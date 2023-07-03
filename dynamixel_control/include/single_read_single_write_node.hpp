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

#ifndef SINGLE_READ_SINGLE_WRITE_NODE_HPP_
#define SINGLE_READ_SINGLE_WRITE_NODE_HPP_

#include <cstdio>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "dynamixel_control_custom_interfaces/msg/set_position.hpp"
#include "dynamixel_control_custom_interfaces/srv/get_position.hpp"


class SingleReadSingleWriteNode : public rclcpp::Node
{
public:
  using SetPosition = dynamixel_control_custom_interfaces::msg::SetPosition;
  using GetPosition = dynamixel_control_custom_interfaces::srv::GetPosition;

  SingleReadSingleWriteNode();
  virtual ~SingleReadSingleWriteNode();

private:
  std::vector<rclcpp::Subscription<SetPosition>::SharedPtr> set_position_subscribers_;
  rclcpp::Service<GetPosition>::SharedPtr get_position_server_;

  int present_position;
};

#endif  // SINGLE_READ_SINGLE_WRITE_NODE_HPP_
