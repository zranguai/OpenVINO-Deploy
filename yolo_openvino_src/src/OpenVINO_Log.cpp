#include "OpenVINO_Log.h"

void Logger::initialize()
{
    // 创建日志文件夹
    createLogFolder(_logFolder);

    // 创建RotatingLogger，每天创建新的日志文件
    Info = spdlog::daily_logger_mt("info_logger", _logFolder + "/Info.log", 0, 0);
    Warn = spdlog::daily_logger_mt("warn_logger", _logFolder + "/Warn.log", 0, 0);
    Error = spdlog::daily_logger_mt("error_logger", _logFolder + "/Error.log", 0, 0);

    //设置日志的级别
    Info->set_level(spdlog::level::info);
    Warn->set_level(spdlog::level::warn);
    Error->set_level(spdlog::level::err);

    //设置刷新频率
    Info->flush_on(spdlog::level::info);
    Warn->flush_on(spdlog::level::warn);
    Error->flush_on(spdlog::level::err);

    //创建定时器，并设置回调函数为删除文件的函数
    std::thread t(&Logger::removeOldLogs, this);
    t.detach();//分离线程，使其在后台运行
}

void Logger::info(const std::string& message) {
    Info->info(message);
}

void Logger::warn(const std::string& message) {
    Warn->warn(message);
}

void Logger::error(const std::string& message) {
    Error->error(message);
}


void Logger::createLogFolder(const std::string logFolder) {
    std::filesystem::create_directories(logFolder);
}

void Logger::Logger::removeOldLogs() {

    while (true) {
        try
        {
            std::this_thread::sleep_for(std::chrono::seconds(10 * 60)); // 每隔10 * 60秒执行一次

            auto now = std::chrono::system_clock::now();
            auto timePoint = now - std::chrono::hours(_iMaxDaysToKeep * 24);
            std::time_t time = std::chrono::system_clock::to_time_t(timePoint);

            std::filesystem::path folderPath(_logFolder);
            for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
                if (std::filesystem::is_directory(entry)) {
                    continue;
                }

                std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(entry);
                std::chrono::system_clock::time_point lastWriteTimePoint = std::chrono::time_point_cast<std::chrono::system_clock::duration>(lastWriteTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                std::time_t lastWriteTimeT = std::chrono::system_clock::to_time_t(lastWriteTimePoint);
                if (lastWriteTimeT < time && entry.path().extension() == ".log") {
                    std::filesystem::remove(entry);
                }
            }
        }
        catch (const std::exception& e)
        {
            std::string msg = "Logger::removeOldLogs::";
            std::string strExcep = e.what();
            msg = msg + strExcep;
            std::cout << msg << std::endl;
        }
    }
}
