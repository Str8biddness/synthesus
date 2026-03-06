// Synthesus 2.0 - main.cpp - IPC stdin/stdout protocol entry point
// Build: bash build.sh --rebuild
// Manual: g++ -std=c++17 -O3 -march=native -o zo_kernel main.cpp \
//   kernel/*.cpp memory/*.cpp reasoning/*.cpp vcu/*.cpp \
//   core/*.cpp automation/*.cpp onnx_bridge/*.cpp \
//   vendor/sqlite3.c -lpthread
#include <iostream>
#include <string>
#include "kernel/thread_pool.hpp"
#include "kernel/message_bus.hpp"
#include "core/hemi_reconciler.hpp"
#include "core/ppbrs_router.hpp"
#include "core/context_memory.hpp"
#include "automation/watchdog.hpp"
int main(int argc, char* argv[]) {
    zo::ThreadPool pool(4);
    zo::MessageBus bus;
    zo::PPBRSRouter router;
    zo::ContextMemory ctx("context.db");
    zo::Watchdog watchdog;
    watchdog.start();
    std::cerr << "[ZO] Synthesus 2.0 kernel ready (stdin IPC)\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "quit" || line == "exit") break;
        ctx.store("last_query", line);
        auto result = router.route(line, ctx.recall("context"));
        std::cout << "{\"r\":\"" << result.response
                  << "\",\"c\":" << result.confidence
                  << ",\"m\":\"" << result.module_used
                  << "\"}" << "\n";
        std::cout.flush();
    }
    watchdog.stop();
    return 0;
}