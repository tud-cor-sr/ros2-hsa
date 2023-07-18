// Copyright 2021 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*******************************************************************************
// This example is written for DYNAMIXEL X(excluding XL-320) and MX(2.0) series with U2D2.
// For other series, please refer to the product eManual and modify the Control Table addresses and other definitions.
// To test this example, please follow the commands below.
//
// Open terminal #1
// $ ros2 run dynamixel_control single_read_single_write_node
//
// Open terminal #2 (run one of below commands at a time)
// $ ros2 topic pub -1 /set_position_motor_21 dynamixel_control_custom_interfaces/SetPosition "{id: 21, position: 1000}"
// $ ros2 service call /get_position dynamixel_control_custom_interfaces/srv/GetPosition "id: 21"
//
// Author: Will Son, Maximilian Stölzle
*******************************************************************************/

#include <cstdio>
#include <memory>
#include <string>

#include "dynamixel_sdk/dynamixel_sdk.h"
#include "dynamixel_control_custom_interfaces/msg/set_position.hpp"
#include "dynamixel_control_custom_interfaces/srv/get_position.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"

#include "single_read_single_write_node.hpp"

// Control table address for X series (except XL-320)
#define ADDR_RETURN_DELAY_TIME 9
#define ADDR_OPERATING_MODE 11
#define ADDR_TORQUE_ENABLE 64
#define ADDR_GOAL_POSITION 116
#define ADDR_PRESENT_POSITION 132

// Protocol version
#define PROTOCOL_VERSION 2.0  // Default Protocol version of DYNAMIXEL X series.

// Default setting
#define BAUDRATE 57600  // Default Baudrate of DYNAMIXEL X series
#define DEVICE_NAME "/dev/ttyUSB0"  // [Linux]: "/dev/ttyUSB*", [Windows]: "COM*"

dynamixel::PortHandler * portHandler;
dynamixel::PacketHandler * packetHandler;

uint8_t dxl_error = 0;
uint32_t goal_position = 0;
int dxl_comm_result = COMM_TX_FAIL;

SingleReadSingleWriteNode::SingleReadSingleWriteNode()
: Node("single_read_single_write_node")
{
  RCLCPP_INFO(this->get_logger(), "Run single read single write node");

  this->declare_parameter("qos_depth", 10);
  int8_t qos_depth = 0;
  this->get_parameter("qos_depth", qos_depth);

  this->declare_parameter("verbose", true);
  bool verbose;
  this->get_parameter("verbose", verbose);

  std::vector<uint8_t> motor_ids{ 21, 22, 23, 24 };
  this->declare_parameter("motor_ids", motor_ids);
  this->get_parameter("motor_ids", motor_ids);

  const auto QOS_RKL10V =
    rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable().durability_volatile();

  // loop over all motor ids and create a subscriber for each
  for (auto motor_id : motor_ids) {
    auto set_position_subscriber =
      this->create_subscription<SetPosition>(
      "set_position_motor_" + std::to_string(motor_id),
      QOS_RKL10V,
      [this, verbose, motor_id](const SetPosition::SharedPtr msg) -> void
      {
        uint8_t dxl_error = 0;

        // Position Value of X series is 4 byte data.
        // For AX & MX(1.0) use 2 byte data(uint16_t) for the Position Value.
        uint32_t goal_position = (unsigned int)msg->position;  // Convert int32 -> uint32

        // Write Goal Position (length : 4 bytes)
        // When writing 2 byte data to AX / MX(1.0), use write2ByteTxRx() instead.
        dxl_comm_result =
        packetHandler->write4ByteTxRx(
          portHandler,
          (uint8_t) msg->id,
          ADDR_GOAL_POSITION,
          goal_position,
          &dxl_error
        );

        if (dxl_comm_result != COMM_SUCCESS) {
          RCLCPP_ERROR(this->get_logger(), "%s", packetHandler->getTxRxResult(dxl_comm_result));
        } else if (dxl_error != 0) {
          RCLCPP_ERROR(this->get_logger(), "%s", packetHandler->getRxPacketError(dxl_error));
        } else {
          if (verbose) {
            RCLCPP_INFO(this->get_logger(), "Set [ID: %d] [Goal Position: %d]", msg->id, msg->position);
          }
        }
      }
      );
    set_position_subscribers_.push_back(set_position_subscriber);
  }


  auto get_present_position =
    [this, verbose](
    const std::shared_ptr<GetPosition::Request> request,
    std::shared_ptr<GetPosition::Response> response) -> void
    {
      // Read Present Position (length : 4 bytes) and Convert uint32 -> int32
      // When reading 2 byte data from AX / MX(1.0), use read2ByteTxRx() instead.
      dxl_comm_result = packetHandler->read4ByteTxRx(
        portHandler,
        (uint8_t) request->id,
        ADDR_PRESENT_POSITION,
        reinterpret_cast<uint32_t *>(&present_position),
        &dxl_error
      );

      if (verbose) {
        RCLCPP_INFO(
          this->get_logger(),
          "Get [ID: %d] [Present Position: %d]",
          request->id,
          present_position
        );
      }

      response->header.stamp = rclcpp::Clock().now();
      response->position = present_position;
    };

  get_position_server_ = create_service<GetPosition>("get_position", get_present_position);
}

SingleReadSingleWriteNode::~SingleReadSingleWriteNode()
{
}

void setupDynamixel(uint8_t dxl_id)
{
  // Set return delay time very low (2 microseconds)
  dxl_comm_result = packetHandler->write1ByteTxRx(
    portHandler,
    dxl_id,
    ADDR_RETURN_DELAY_TIME,
    1,
    &dxl_error
  );

  if (dxl_comm_result != COMM_SUCCESS) {
    RCLCPP_ERROR(rclcpp::get_logger("single_read_single_write"), "Failed to set the return delay time.");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("single_read_single_write"), "Succeeded to set the return delay time.");
  }

  // Use Extended Position Control Mode
  dxl_comm_result = packetHandler->write1ByteTxRx(
    portHandler,
    dxl_id,
    ADDR_OPERATING_MODE,
    4,  // set to multiturn mode (3 would be standard position control)
    &dxl_error
  );

  if (dxl_comm_result != COMM_SUCCESS) {
    RCLCPP_ERROR(rclcpp::get_logger("single_read_single_write_node"), "Failed to set Extended Position Control Mode.");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("single_read_single_write_node"), "Succeeded to set Extended Position Control Mode.");
  }

  // Enable Torque of DYNAMIXEL
  dxl_comm_result = packetHandler->write1ByteTxRx(
    portHandler,
    dxl_id,
    ADDR_TORQUE_ENABLE,
    1,
    &dxl_error
  );

  if (dxl_comm_result != COMM_SUCCESS) {
    RCLCPP_ERROR(rclcpp::get_logger("single_read_single_write_node"), "Failed to enable torque.");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("single_read_single_write_node"), "Succeeded to enable torque.");
  }
}

int main(int argc, char * argv[])
{
  portHandler = dynamixel::PortHandler::getPortHandler(DEVICE_NAME);
  packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  // Open Serial Port
  dxl_comm_result = portHandler->openPort();
  if (dxl_comm_result == false) {
    RCLCPP_ERROR(rclcpp::get_logger("single_read_single_write_node"), "Failed to open the port!");
    return -1;
  } else {
    RCLCPP_INFO(rclcpp::get_logger("single_read_single_write_node"), "Succeeded to open the port.");
  }

  // Set the baudrate of the serial port (use DYNAMIXEL Baudrate)
  dxl_comm_result = portHandler->setBaudRate(BAUDRATE);
  if (dxl_comm_result == false) {
    RCLCPP_ERROR(rclcpp::get_logger("single_read_single_write_node"), "Failed to set the baudrate!");
    return -1;
  } else {
    RCLCPP_INFO(rclcpp::get_logger("single_read_single_write_node"), "Succeeded to set the baudrate.");
  }

  setupDynamixel(BROADCAST_ID);

  rclcpp::init(argc, argv);

  auto node = std::make_shared<SingleReadSingleWriteNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  // Disable Torque of DYNAMIXEL
  packetHandler->write1ByteTxRx(
    portHandler,
    BROADCAST_ID,
    ADDR_TORQUE_ENABLE,
    0,
    &dxl_error
  );

  return 0;
}
