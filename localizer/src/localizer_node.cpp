#include <queue>
#include <mutex>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
#include "interface/srv/relocalize.hpp"
#include "interface/srv/is_valid.hpp"
#include <yaml-cpp/yaml.h>
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

using namespace std::chrono_literals;

struct NodeConfig {
  std::string cloud_topic = "/fastlio2/body_cloud";
  std::string odom_topic = "/fastlio2/lio_odom";
  std::string map_frame = "map";
  std::string local_frame = "lidar";
  std::string base_frame = "base_link";
  std::string initialpose_topic = "/initialpose";
  std::string map_pcd_path = "";
  bool auto_load_map = true;
  double update_hz = 1.0;     // localization update
  double tf_hz = 30.0;        // TF broadcast update
};

struct NodeState
{
    std::mutex message_mutex;
    std::mutex service_mutex;
    std::mutex tf_mutex;

    bool message_received = false;
    bool service_received = false;
    bool localize_success = false;
    bool localizing = false;

    rclcpp::Time last_localize_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
    builtin_interfaces::msg::Time last_message_time;

    CloudType::Ptr last_cloud = std::make_shared<CloudType>();

    // odom/local frame pose
    M3D last_r = M3D::Identity();
    V3D last_t = V3D::Zero();

    // map -> local_frame
    M3D last_offset_r = M3D::Identity();
    V3D last_offset_t = V3D::Zero();

    M4F initial_guess = M4F::Identity();
};

class LocalizerNode : public rclcpp::Node
{
public:
    LocalizerNode() : Node("localizer_node")
    {
        RCLCPP_INFO(this->get_logger(), "Localizer Node Started");

        loadParameters();

        rclcpp::QoS qos = rclcpp::QoS(10);
        m_cloud_sub.subscribe(this, m_config.cloud_topic, qos.get_rmw_qos_profile());
        m_odom_sub.subscribe(this, m_config.odom_topic, qos.get_rmw_qos_profile());

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_sync = std::make_shared<
            message_filters::Synchronizer<
                message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::PointCloud2,
                    nav_msgs::msg::Odometry>>>(
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::PointCloud2,
                nav_msgs::msg::Odometry>(10),
            m_cloud_sub, m_odom_sub);

        m_sync->setAgePenalty(0.1);
        m_sync->registerCallback(
            std::bind(&LocalizerNode::syncCB, this,
                      std::placeholders::_1, std::placeholders::_2));

        m_localizer = std::make_shared<ICPLocalizer>(m_localizer_config);

        m_reloc_srv = this->create_service<interface::srv::Relocalize>(
            "relocalize",
            std::bind(&LocalizerNode::relocCB, this,
                      std::placeholders::_1, std::placeholders::_2));

        m_reloc_check_srv = this->create_service<interface::srv::IsValid>(
            "relocalize_check",
            std::bind(&LocalizerNode::relocCheckCB, this,
                      std::placeholders::_1, std::placeholders::_2));

        m_map_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);

        m_initialpose_sub =
            this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                m_config.initialpose_topic, 10,
                std::bind(&LocalizerNode::initialPoseCB, this, std::placeholders::_1));

        // TF broadcast timer (high rate, lightweight)
        auto tf_period = std::chrono::duration<double>(1.0 / std::max(1.0, m_config.tf_hz));
        m_tf_timer = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(tf_period),
            std::bind(&LocalizerNode::tfTimerCB, this));

        // localization timer (low rate, heavyweight)
        auto loc_period = std::chrono::duration<double>(1.0 / std::max(0.1, m_config.update_hz));
        m_localize_timer = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(loc_period),
            std::bind(&LocalizerNode::localizeTimerCB, this));

        if (m_config.auto_load_map && !m_config.map_pcd_path.empty()) {
            if (std::filesystem::exists(m_config.map_pcd_path)) {
                m_map_loaded = m_localizer->loadMap(m_config.map_pcd_path);
                if (m_map_loaded) {
                    RCLCPP_INFO(this->get_logger(), "Loaded map from: %s", m_config.map_pcd_path.c_str());
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Failed to load map from: %s", m_config.map_pcd_path.c_str());
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "Map file not found: %s", m_config.map_pcd_path.c_str());
            }
        }
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_config.cloud_topic = config["cloud_topic"].as<std::string>();
        m_config.odom_topic = config["odom_topic"].as<std::string>();
        m_config.map_frame = config["map_frame"].as<std::string>();
        m_config.local_frame = config["local_frame"].as<std::string>();

        if (config["base_frame"]) {
            m_config.base_frame = config["base_frame"].as<std::string>();
        }

        m_config.update_hz = config["update_hz"].as<double>();

        if (config["tf_hz"]) {
            m_config.tf_hz = config["tf_hz"].as<double>();
        }

        m_localizer_config.rough_scan_resolution = config["rough_scan_resolution"].as<double>();
        m_localizer_config.rough_map_resolution = config["rough_map_resolution"].as<double>();
        m_localizer_config.rough_max_iteration = config["rough_max_iteration"].as<int>();
        m_localizer_config.rough_score_thresh = config["rough_score_thresh"].as<double>();

        m_localizer_config.refine_scan_resolution = config["refine_scan_resolution"].as<double>();
        m_localizer_config.refine_map_resolution = config["refine_map_resolution"].as<double>();
        m_localizer_config.refine_max_iteration = config["refine_max_iteration"].as<int>();
        m_localizer_config.refine_score_thresh = config["refine_score_thresh"].as<double>();

        m_config.initialpose_topic = config["initialpose_topic"].as<std::string>();
        m_config.map_pcd_path = config["map_pcd_path"].as<std::string>();
        m_config.auto_load_map = config["auto_load_map"].as<bool>();
    }

    void syncCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
                const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
        std::lock_guard<std::mutex> lock(m_state.message_mutex);

        pcl::fromROSMsg(*cloud_msg, *m_state.last_cloud);

        // odom_msg->pose.pose は
        //   odom_msg->header.frame_id  ->  odom_msg->child_frame_id
        // の姿勢
        m_state.last_r =
            Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                            odom_msg->pose.pose.orientation.x,
                            odom_msg->pose.pose.orientation.y,
                            odom_msg->pose.pose.orientation.z)
                .toRotationMatrix();

        m_state.last_t = V3D(odom_msg->pose.pose.position.x,
                            odom_msg->pose.pose.position.y,
                            odom_msg->pose.pose.position.z);

        m_state.last_message_time = cloud_msg->header.stamp;

        if (!m_state.message_received)
        {
            m_state.message_received = true;
            m_config.local_frame = odom_msg->header.frame_id;

            RCLCPP_INFO(this->get_logger(),
                        "First odom frame_id detected as local_frame: %s",
                        m_config.local_frame.c_str());
            RCLCPP_INFO(this->get_logger(),
                        "Odometry child_frame_id: %s",
                        odom_msg->child_frame_id.c_str());
        }
    }

    void tfTimerCB()
    {
        if (!m_state.message_received)
            return;

        sendBroadCastTF();
    }

    void localizeTimerCB()
    {
        if (!m_state.message_received)
            return;

        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            if (m_state.localizing) {
                return;
            }
            m_state.localizing = true;
        }

        doLocalization();

        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            m_state.localizing = false;
        }
    }

    void doLocalization()
    {
        bool has_pending_relocalize = false;
        M4F initial_guess = M4F::Identity();

        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            has_pending_relocalize = m_state.service_received;
            if (has_pending_relocalize) {
                initial_guess = m_state.initial_guess;
            }
        }

        M3D current_local_r;
        V3D current_local_t;
        CloudType::Ptr current_cloud = std::make_shared<CloudType>();

        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            current_local_r = m_state.last_r;
            current_local_t = m_state.last_t;
            *current_cloud = *m_state.last_cloud;
        }

        if (!has_pending_relocalize)
        {
            std::lock_guard<std::mutex> tf_lock(m_state.tf_mutex);
            initial_guess.block<3, 3>(0, 0) =
                (m_state.last_offset_r * current_local_r).cast<float>();
            initial_guess.block<3, 1>(0, 3) =
                (m_state.last_offset_r * current_local_t + m_state.last_offset_t).cast<float>();
        }

        if (has_pending_relocalize)
        {
            RCLCPP_WARN(this->get_logger(), "FORCE JUMP TO INITIAL GUESS");

            M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
            V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();

            {
                std::lock_guard<std::mutex> tf_lock(m_state.tf_mutex);
                m_state.last_offset_r = map_body_r;
                m_state.last_offset_t = map_body_t;
            }
        }



        m_localizer->setInput(current_cloud);


        bool result = m_localizer->align(initial_guess);


        if (result)
        {
            //RCLCPP_INFO(this->get_logger(), "Relocalization SUCCESS");

            M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
            V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();

            M3D new_offset_r = map_body_r * current_local_r.transpose();
            V3D new_offset_t =
                -map_body_r * current_local_r.transpose() * current_local_t + map_body_t;

            {
                std::lock_guard<std::mutex> tf_lock(m_state.tf_mutex);
                m_state.last_offset_r = new_offset_r;
                m_state.last_offset_t = new_offset_t;
            }

            if (has_pending_relocalize)
            {
                std::lock_guard<std::mutex> lock(m_state.service_mutex);
                m_state.localize_success = true;
                m_state.service_received = false;
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Relocalization FAILED");
        }

        publishMapCloud();
    }

    void sendBroadCastTF()
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = m_config.map_frame;
        transformStamped.child_frame_id = m_config.local_frame;

        M3D r;
        V3D t;
        builtin_interfaces::msg::Time stamp;

        {
            std::lock_guard<std::mutex> tf_lock(m_state.tf_mutex);
            r = m_state.last_offset_r;
            t = m_state.last_offset_t;
        }

        {
            std::lock_guard<std::mutex> msg_lock(m_state.message_mutex);
            stamp = m_state.last_message_time;
        }

        transformStamped.header.stamp = stamp;

        Eigen::Quaterniond q(r);
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        m_tf_broadcaster->sendTransform(transformStamped);
    }

    void relocCB(const std::shared_ptr<interface::srv::Relocalize::Request> request,
                 std::shared_ptr<interface::srv::Relocalize::Response> response)
    {
        std::string pcd_path = request->pcd_path;
        float x = request->x;
        float y = request->y;
        float z = request->z;
        float yaw = request->yaw;
        float roll = request->roll;
        float pitch = request->pitch;

        if (!std::filesystem::exists(pcd_path))
        {
            response->success = false;
            response->message = "pcd file not found";
            return;
        }

        Eigen::AngleAxisd yaw_angle = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd roll_angle = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());

        bool load_flag = m_localizer->loadMap(pcd_path);
        if (!load_flag)
        {
            response->success = false;
            response->message = "load map failed";
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            m_state.initial_guess.setIdentity();
            m_state.initial_guess.block<3, 3>(0, 0) =
                (yaw_angle * roll_angle * pitch_angle).toRotationMatrix().cast<float>();
            m_state.initial_guess.block<3, 1>(0, 3) = V3F(x, y, z);
            m_state.service_received = true;
            m_state.localize_success = false;
        }

        response->success = true;
        response->message = "relocalize success";
    }

    void relocCheckCB(const std::shared_ptr<interface::srv::IsValid::Request> request,
                      std::shared_ptr<interface::srv::IsValid::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_state.service_mutex);
        if (request->code == 1)
            response->valid = true;
        else
            response->valid = m_state.localize_success;
    }

    void publishMapCloud()
    {
        if (m_map_cloud_pub->get_subscription_count() < 1)
            return;

        CloudType::Ptr map_cloud = m_localizer->refineMap();
        if (map_cloud->size() < 1)
            return;

        sensor_msgs::msg::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*map_cloud, map_cloud_msg);
        map_cloud_msg.header.frame_id = m_config.map_frame;
        map_cloud_msg.header.stamp = this->get_clock()->now();
        m_map_cloud_pub->publish(map_cloud_msg);
    }

    void initialPoseCB(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (!m_map_loaded) {
            if (m_config.map_pcd_path.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Map is not loaded and map_pcd_path is empty.");
                return;
            }
            if (!std::filesystem::exists(m_config.map_pcd_path)) {
                RCLCPP_ERROR(this->get_logger(), "Map file not found: %s", m_config.map_pcd_path.c_str());
                return;
            }
            m_map_loaded = m_localizer->loadMap(m_config.map_pcd_path);
            if (!m_map_loaded) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load map from: %s", m_config.map_pcd_path.c_str());
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Loaded map from: %s", m_config.map_pcd_path.c_str());
        }

        // RVizの /initialpose は map -> base_frame として解釈する
        const auto &p = msg->pose.pose.position;
        const auto &q = msg->pose.pose.orientation;

        M3D map_base_r = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
        V3D map_base_t(p.x, p.y, p.z);

        // 現在の odom(local_frame) -> base_frame(child_frame_id想定) を取得
        M3D local_base_r;
        V3D local_base_t;

        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            if (!m_state.message_received) {
                RCLCPP_WARN(this->get_logger(),
                            "No odom received yet. Cannot convert map->%s to map->%s.",
                            m_config.base_frame.c_str(),
                            m_config.local_frame.c_str());
                return;
            }

            local_base_r = m_state.last_r;
            local_base_t = m_state.last_t;
        }

        // 変換:
        // map->local = map->base * inv(local->base)
        M3D map_local_r = map_base_r * local_base_r.transpose();
        V3D map_local_t = map_base_t - map_local_r * local_base_t;

        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            m_state.initial_guess.setIdentity();
            m_state.initial_guess.block<3, 3>(0, 0) = map_local_r.cast<float>();
            m_state.initial_guess.block<3, 1>(0, 3) = map_local_t.cast<float>();
            m_state.service_received = true;
            m_state.localize_success = false;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Received /initialpose as map->%s, converted to map->%s: x=%.3f y=%.3f z=%.3f",
                    m_config.base_frame.c_str(),
                    m_config.local_frame.c_str(),
                    map_local_t.x(), map_local_t.y(), map_local_t.z());
    }

private:
    NodeConfig m_config;
    NodeState m_state;

    ICPConfig m_localizer_config;
    std::shared_ptr<ICPLocalizer> m_localizer;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> m_cloud_sub;
    message_filters::Subscriber<nav_msgs::msg::Odometry> m_odom_sub;
    std::shared_ptr<
        message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::PointCloud2,
                nav_msgs::msg::Odometry>>> m_sync;

    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

    rclcpp::Service<interface::srv::Relocalize>::SharedPtr m_reloc_srv;
    rclcpp::Service<interface::srv::IsValid>::SharedPtr m_reloc_check_srv;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_cloud_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_initialpose_sub;

    rclcpp::TimerBase::SharedPtr m_tf_timer;
    rclcpp::TimerBase::SharedPtr m_localize_timer;

    bool m_map_loaded = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizerNode>());
    rclcpp::shutdown();
    return 0;
}