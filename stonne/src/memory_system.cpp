//dramsim3

#include "memory_system.h"

namespace dramsim3 {
MemorySystem::MemorySystem(const std::string &config_file,
                           const std::string &output_dir,
                           std::function<void(uint64_t)> read_callback,
                           std::function<void(uint64_t)> write_callback)
    : config_(new Config(config_file, output_dir)) {
    // TODO: ideal memory type?
	//std::cout << "MemorySystem IN" << std::endl;
    if (config_->IsHMC()) {
        dram_system_ = new HMCMemorySystem(*config_, output_dir, read_callback,
                                           write_callback);
    } else {
        dram_system_ = new JedecDRAMSystem(*config_, output_dir, read_callback,
                                           write_callback);
    }
	//std::cout << "MemorySystem OUT" << std::endl;
}

MemorySystem::~MemorySystem() {
    delete (dram_system_);
    delete (config_);
}

void MemorySystem::ClockTick() { 
	dram_system_->ClockTick(); 
	std::queue<int> delay;
	delay = dram_system_->ReturnDelay();
	while (!delay.empty())
	{
		this->read_delay.push(delay.front());
		delay.pop();
	}
	//dram_system_->FreeDelayQueue();
}

std::queue<int> MemorySystem::ReturnDelay() {
	std::queue<int> delay;
	while (!this->read_delay.empty())
	{
		delay.push(this->read_delay.front());
		this->read_delay.pop();
	}
	return delay;
}

void MemorySystem::FreeDelayQueue() {
	while (!this->read_delay.empty()) {
		this->read_delay.pop();
	}
}


double MemorySystem::GetTCK() const { return config_->tCK; }

int MemorySystem::GetBusBits() const { return config_->bus_width; }

int MemorySystem::GetBurstLength() const { return config_->BL; }

int MemorySystem::GetQueueSize() const { return config_->trans_queue_size; }

void MemorySystem::RegisterCallbacks(
    std::function<void(uint64_t)> read_callback,
    std::function<void(uint64_t)> write_callback) {
    dram_system_->RegisterCallbacks(read_callback, write_callback);
}

bool MemorySystem::WillAcceptTransaction(uint64_t hex_addr,
                                         bool is_write) const {
    return dram_system_->WillAcceptTransaction(hex_addr, is_write);
}

bool MemorySystem::AddTransaction(uint64_t hex_addr, bool is_write,std::string addr_str) {
    return dram_system_->AddTransaction(hex_addr, is_write, addr_str);
}

 
void MemorySystem::PrintStats() const { dram_system_->PrintStats(); }

void MemorySystem::ResetStats() { dram_system_->ResetStats(); }

MemorySystem* GetMemorySystem(const std::string &config_file, const std::string &output_dir,
                 std::function<void(uint64_t)> read_callback,
                 std::function<void(uint64_t)> write_callback) {
    return new MemorySystem(config_file, output_dir, read_callback, write_callback);
}
}  // namespace dramsim3

// This function can be used by autoconf AC_CHECK_LIB since
// apparently it can't detect C++ functions.
// Basically just an entry in the symbol table
extern "C" {
void libdramsim3_is_present(void) { ; }
}
