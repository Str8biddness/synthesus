#pragma once
// Synthesus 2.0 Phase 7 - VCU Base (Virtual Cortex Unit abstract base)
#include <string>
#include <functional>
namespace zo {
struct VCUInput { std::string data; std::string context; float urgency{0.5f}; };
struct VCUOutput { std::string result; float confidence; std::string vcu_id; bool handled{false}; };
class VCUBase {
public:
    virtual ~VCUBase() = default;
    virtual std::string id() const = 0;
    virtual VCUOutput process(const VCUInput& input) = 0;
    virtual bool can_handle(const VCUInput& input) const = 0;
    virtual void on_tick(uint64_t tick_ms) {}
    virtual void reset() {}
    bool enabled{true};
    float priority{1.0f};
};
} // namespace zo
