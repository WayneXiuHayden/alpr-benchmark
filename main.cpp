#include <alpr.h>
#include <alprstream.h>
#include <string>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <gflags/gflags.h>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>

DEFINE_bool(use_gpu, false, "Use GPU for processing");
DEFINE_uint32(batch_size, 10, "Batch size for processing");
DEFINE_uint32(stream_queue_size, 200, "Stream queue size");

void video_input(alpr::AlprStream* alpr_stream, const std::string& video_file_path, std::atomic<bool>& video_connected) {
    alpr_stream->connect_video_file(video_file_path, 0);
    video_connected = true;
}

void test_stream_api(bool use_gpu, int batch_size, int stream_queue_size) {

    std::cout << "Test stream API" << std::endl;
    std::ifstream alpr_key_file("/mnt/data/config/lpr_key.txt");
    std::string alpr_key((std::istreambuf_iterator<char>(alpr_key_file)),
                          std::istreambuf_iterator<char>());

    std::cout << "Loading ALPR config..." << std::endl;
    alpr::Alpr alpr(
        /* country */ "us",
        "/mnt/data/config/openalpr.gpu.conf",
        "/usr/share/openalpr/runtime_data", 
        alpr_key,
        use_gpu ? alpr::AlprHardwareAcceleration::ALPR_NVIDIA_GPU : alpr::AlprHardwareAcceleration::ALPR_CPU,
        /* GPU ID */ 0,
        /* batch size */ batch_size);

    if (alpr.isLoaded() == false){
        std::cerr << "Error loading OpenALPR" << std::endl;
        return;
    }
    alpr.setTopN(5);
    alpr.setDetectRegion(false);
    alpr.setDetectVehicles(false, false);
    
    // const int VIDEO_BUFFER_SIZE = 200;
    const int USE_MOTION_DETECTION = 1;
    alpr::AlprStream alpr_stream(
        /* queue size */ stream_queue_size,
        /* use motion detection */ USE_MOTION_DETECTION
    );
    // TODO: this will crash with following error message:
    // terminate called after throwing an instance of 'std::runtime_error'
    //   what():  CUDA failed with error invalid device ordinal at /debout/openalpr_5.0.0.orig/openalprgpu/src/cuda/alprgpusupport_cudaimpl.cpp:195
    // if (use_gpu) {
    //     alpr_stream.set_gpu_async(0);
    // }

    std::cout << "Connecting video file..." << std::endl;
    // std::string video_file_path = "../data/videos/720p.mp4";
    std::string video_file_path = "../data/videos/video_0.mp4";
    // alpr_stream.connect_video_file(video_file_path, 0);
    std::atomic<bool> video_connected(false);
    std::thread video_thread(video_input, &alpr_stream, video_file_path, std::ref(video_connected));
    while (!video_connected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Processing frames..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    std::vector<std::vector<alpr::AlprGroupResult>> groups_results;
    while (alpr_stream.video_file_active())
    {
        if (alpr_stream.get_queue_size() <= 0)
        {
            continue;
        }

        std::vector<alpr::RecognizedFrame> frame_result = alpr_stream.process_batch(&alpr); // process_batch
        if (i == 0)
        {
            start = std::chrono::high_resolution_clock::now();
            i++;
        }

        groups_results.emplace_back(alpr_stream.pop_completed_groups());
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken by function: " << duration.count()/1e6 << " seconds" << std::endl;

    video_thread.join();

    std::ofstream out("test.json"/*, std::ios::app*/);
    for (const auto& group_result : groups_results)
    {
        for (alpr::AlprGroupResult result : group_result) {
            out << result.toJson() << std::endl;
        }
    }
    out.close();
}

void test_batch_image_api(bool use_gpu, int batch_size) {

    std::cout << "Test batch image API" << std::endl;
    std::ifstream alpr_key_file("/mnt/data/config/lpr_key.txt");
    std::string alpr_key((std::istreambuf_iterator<char>(alpr_key_file)),
                          std::istreambuf_iterator<char>());

    std::cout << "Loading ALPR config..." << std::endl;
    alpr::Alpr alpr(
        /* country */ "us",
        "/mnt/data/config/openalpr.gpu.conf",
        "/usr/share/openalpr/runtime_data", 
        alpr_key,
        use_gpu ? alpr::AlprHardwareAcceleration::ALPR_NVIDIA_GPU : alpr::AlprHardwareAcceleration::ALPR_CPU,
        /* GPU ID */ 0,
        /* batch size */ batch_size);

    if (alpr.isLoaded() == false){
        std::cerr << "Error loading OpenALPR" << std::endl;
        return;
    }
    alpr.setTopN(5);
    alpr.setDetectRegion(false);
    alpr.setDetectVehicles(false, false);

    // Pre-read all images
    std::vector<cv::Mat> alpr_mats;
    std::vector<unsigned char*> alpr_data;

    for (const auto& entry : boost::filesystem::directory_iterator("../data/images")) {
        std::string image_path = entry.path().string();
        alpr_mats.push_back(cv::imread(image_path));
        alpr_data.push_back(alpr_mats.back().data);
    }

    // Process images in batches
    size_t num_images = alpr_mats.size();
    size_t num_batches = (num_images + batch_size - 1) / batch_size;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        size_t start_idx = batch_idx * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, num_images);

        std::vector<alpr::AlprResults> batch_results;
        std::vector<unsigned char*> batch_data;
        std::vector<std::vector<alpr::AlprRegionOfInterest>> batch_rois;

        for (size_t i = start_idx; i < end_idx; i++) {
            batch_data.push_back(alpr_data[i]);
            batch_rois.push_back(std::vector<alpr::AlprRegionOfInterest>());
        }

        size_t elem_size = alpr_mats[start_idx].elemSize();
        int width = alpr_mats[start_idx].cols;
        int height = alpr_mats[start_idx].rows;

        batch_results = alpr.recognize_batch(batch_data.data(), elem_size, width, height, batch_data.size(), batch_rois);
        assert(batch_results.size() == batch_data.size());
        // std::cout << "Batch " << batch_idx + 1 << ": " << batch_results.size() << " images processed" << std::endl;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken by function: " << duration.count()/1e6 << " seconds" << std::endl;
}

void process_frames(std::unique_ptr<alpr::Alpr> alpr, alpr::AlprStream* alpr_stream, 
                    std::vector<std::vector<alpr::AlprGroupResult>>& groups_results,
                    std::atomic<bool>& processing_done, std::mutex& results_mutex) {

    std::vector<std::vector<alpr::AlprGroupResult>> local_results;

    while (alpr_stream->video_file_active() || alpr_stream->get_queue_size() > 0) {
        if (alpr_stream->get_queue_size() <= 0) {
            continue;
        }

        std::vector<alpr::RecognizedFrame> frame_result = alpr_stream->process_batch(alpr.get());
        
        local_results.emplace_back(alpr_stream->pop_completed_groups());

        if (local_results.size() > 10) {
            std::lock_guard<std::mutex> lock(results_mutex);
            groups_results.insert(groups_results.end(), local_results.begin(), local_results.end());
            local_results.clear();
        }
    }

    {
        std::lock_guard<std::mutex> lock(results_mutex);
        groups_results.insert(groups_results.end(), local_results.begin(), local_results.end());
    }

    processing_done = true;
}

void test_stream_api_thread(bool use_gpu, int batch_size, int stream_queue_size) {
    std::cout << "Test stream API - threaded with GPU optimizations" << std::endl;

    const int USE_MOTION_DETECTION = 1;
    alpr::AlprStream alpr_stream(stream_queue_size, USE_MOTION_DETECTION);

    std::string video_file_path = "../data/videos/video_0.mp4";
    std::atomic<bool> video_connected(false);
    std::thread video_thread(video_input, &alpr_stream, video_file_path, std::ref(video_connected));

    while (!video_connected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Processing frames..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<alpr::AlprGroupResult>> groups_results;
    std::atomic<bool> processing_done(false);
    std::vector<std::thread> processing_threads;
    std::mutex results_mutex;

    auto create_alpr = [&]() {
        return std::make_unique<alpr::Alpr>(
            "us",
            "/mnt/data/config/openalpr.gpu.conf",
            "/usr/share/openalpr/runtime_data", 
            "/mnt/data/config/lpr_key.txt",
            use_gpu ? alpr::AlprHardwareAcceleration::ALPR_NVIDIA_GPU : alpr::AlprHardwareAcceleration::ALPR_CPU,
            0,
            batch_size);
    };

    int num_threads = std::min(2, (int)std::thread::hardware_concurrency());
    for (int i = 0; i < num_threads; ++i) {
        auto alpr = create_alpr();
        if (!alpr->isLoaded()) {
            std::cerr << "Error loading OpenALPR for thread " << i << std::endl;
            return;
        }
        alpr->setTopN(5);
        alpr->setDetectRegion(false);
        alpr->setDetectVehicles(false, false);

        processing_threads.emplace_back(process_frames, std::move(alpr), &alpr_stream, 
                                        std::ref(groups_results), std::ref(processing_done), 
                                        std::ref(results_mutex));
    }

    for (auto& thread : processing_threads) {
        thread.join();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken by function: " << duration.count()/1e6 << " seconds" << std::endl;

    video_thread.join();

    std::ofstream out("test-thread.json");
    for (const auto& group_result : groups_results) {
        for (alpr::AlprGroupResult result : group_result) {
            out << result.toJson() << std::endl;
        }
    }
    out.close();
}





int main(int argc, char *argv[]) {
    
    std::cout << "Testing the OpenALPR library." << std::endl;

    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_use_gpu) std::cout << "Using GPU" << std::endl; 
    else std::cout << "Using CPU" << std::endl;

    std::cout << "===================================\n";
    test_stream_api(FLAGS_use_gpu, FLAGS_batch_size, FLAGS_stream_queue_size);

    std::cout << "===================================\n";
    test_batch_image_api(FLAGS_use_gpu, FLAGS_batch_size);

    std::cout << "===================================\n";
    test_stream_api_thread(FLAGS_use_gpu, FLAGS_batch_size, FLAGS_stream_queue_size);

    return 0;
}
