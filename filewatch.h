#pragma once

#include <atomic>
#include <filesystem>
#include <chrono>
#include <thread>
#include <string>
#include <functional>

namespace filewatch 
{
    enum class EventKind
    {
        Created,
        Deleted,
        Modified
    };

    class FileWatcher 
    {
    public:
        using Callback = std::function<void(const std::string&, EventKind)>;

        FileWatcher(
            const std::string& path,
            const std::chrono::milliseconds poll_interval): 
            m_path(path), 
            m_poll_interval(poll_interval), 
            m_path_exists(std::filesystem::exists(path)) {
            
            if (m_path_exists) {
                m_last_write_time = std::filesystem::last_write_time(path);
            }
            else {
                m_last_write_time = std::filesystem::file_time_type::min();
            }
        }
        
        void start(Callback callback) 
        {
            while (!m_stop_requested) 
            {
                std::this_thread::sleep_for(m_poll_interval);

                if (!m_path_exists && std::filesystem::exists(m_path)) {
                    m_path_exists = true;
                    m_last_write_time = std::filesystem::last_write_time(m_path);
                    callback(m_path, EventKind::Created);
                }
                else if (m_path_exists && !std::filesystem::exists(m_path)) {
                    m_path_exists = false;
                    callback(m_path, EventKind::Deleted);
                }
                else if (m_last_write_time != std::filesystem::last_write_time(m_path)) {
                    m_last_write_time = std::filesystem::last_write_time(m_path);
                    callback(m_path, EventKind::Modified);
                }
            }
        }

        void stop() 
        { 
            m_stop_requested = true;  
        }

    private:
        std::string m_path;
        std::chrono::milliseconds m_poll_interval;
        bool m_path_exists;
        std::filesystem::file_time_type m_last_write_time;
        std::atomic_bool m_stop_requested = false;
    };
}
