#pragma once
#ifndef __TENSORRT_LOG_H__
#define __TENSORRT_LOG_H__

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>

class Logger {
public:
    static Logger& getIntance() {
        static Logger instance;
        return instance;
    }
public:
    // 初始化
    void initialize();

    // 输出Info级别的日志
    // ...
    void info(const std::string& message);

    // 输出Warn级别的日志
    // ...
    void warn(const std::string& message);

    // 输出Error级别的日志
    // ...
    void error(const std::string& message);

private:
    //私有构造函数，确保只能通过getInstance()获取实例
    Logger() {
        initialize();
    };

    void createLogFolder(const std::string logFolder);//创建文件路径
    void removeOldLogs();//移除旧文件

    Logger(const Logger&) = delete;  // 禁止拷贝构造函数
    Logger& operator=(const Logger&) = delete;  // 禁止赋值运算符

private:
    // 其他私有成员变量和函数
    // ...
    std::shared_ptr<spdlog::logger> Info;
    std::shared_ptr<spdlog::logger> Warn;
    std::shared_ptr<spdlog::logger> Error;

    std::string _logFolder = "./DeepLearning/Log";//日志存储路径
    int _iMaxDaysToKeep = 1;//日志最大保存天数
};


#endif