#include "rclcpp/rclcpp.hpp"
#include "tm_msgs/srv/set_positions.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"

#include <chrono>
#include <memory>

using namespace std::chrono_literals;

class CobotCommander : public rclcpp::Node
{
public:
    CobotCommander()
        : Node("cobot_commander"),
          target_received_(false),
          success_published_(false)
    {
        // Create service client
        client_ = this->create_client<tm_msgs::srv::SetPositions>("set_positions");

        // Create subscribers
        obj_coords_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/obj_coords", 10,
            std::bind(&CobotCommander::obj_coords_callback, this, std::placeholders::_1));

        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&CobotCommander::joint_states_callback, this, std::placeholders::_1));

        // Create publisher
        success_pub_ = this->create_publisher<std_msgs::msg::Bool>("/cobot_response", 10);

        RCLCPP_INFO(this->get_logger(), "CobotCommander node started.");
    }

private:
    void obj_coords_callback(const geometry_msgs::msg::Point::SharedPtr msg)
    {
        if (!client_->wait_for_service(3s))
        {
            RCLCPP_ERROR(this->get_logger(), "Service 'set_positions' not available.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received target position: x=%.3f, y=%.3f, z=%.3f", msg->x, msg->y, msg->z);

        auto request = std::make_shared<tm_msgs::srv::SetPositions::Request>();
        request->motion_type = tm_msgs::srv::SetPositions::Request::PTP_T;
        request->positions = {
            msg->x / 1000.0, // x in meters
            msg->y / 1000.0, // y in meters
            msg->z / 1000.0, // z in meters
            1.57,            // roll
            0.0,             // pitch
            0.0              // yaw
        };
        request->velocity = 1.0; // rad/s
        request->acc_time = 0.2; // seconds
        request->blend_percentage = 10;
        request->fine_goal = false;

        auto future = client_->async_send_request(request);

        // Set the flag to start monitoring joint velocities
        target_received_ = true;
        success_published_ = false;

        // Optionally wait for service call response
        auto status = rclcpp::spin_until_future_complete(this->shared_from_this(), future, 5s);
        if (status == rclcpp::FutureReturnCode::SUCCESS)
        {
            if (future.get()->ok)
                RCLCPP_INFO(this->get_logger(), "Motion command accepted!");
            else
                RCLCPP_WARN(this->get_logger(), "Motion command rejected by cobot.");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to call set_positions service.");
        }
    }

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (!target_received_ || success_published_)
        {
            return;
        }

        // Check if all joint velocities are close to zero
        double velocity_tolerance = 0.01; // rad/s
        bool all_stopped = true;
        for (auto v : msg->velocity)
        {
            if (std::abs(v) > velocity_tolerance)
            {
                all_stopped = false;
                break;
            }
        }

        if (all_stopped)
        {
            RCLCPP_INFO(this->get_logger(), "Cobot reached the target position (joint velocities ~ 0)");

            std_msgs::msg::Bool success_msg;
            success_msg.data = true;
            success_pub_->publish(success_msg);

            success_published_ = true;
            target_received_ = false; // Reset for the next target
        }
    }

    // Members
    rclcpp::Client<tm_msgs::srv::SetPositions>::SharedPtr client_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr obj_coords_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr success_pub_;

    bool target_received_;
    bool success_published_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CobotCommander>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
