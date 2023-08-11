#ifndef __CPU_H
#define __CPU_H

#include <fstream>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <bitset>
#include "memory_system.h"
#include "cacheL2.h"
#include "cacheCTR.h"
#include "cacheBMT.h"
#include "cacheMAC.h"
#include "../ext/headers/INIReader.h"

using namespace std;
 namespace dramsim3 {

	class CPU {
	public:

		CPU(const std::string& config_file, const std::string& output_dir)
			: memory_system_(
				config_file, output_dir,
				std::bind(&CPU::ReadCallBack, this, std::placeholders::_1),
				std::bind(&CPU::WriteCallBack, this, std::placeholders::_1)),
			clk_(0),
			completeTime(0),
			l2(new cacheL2(config_file)),
			ctr_cache(new cacheCTR(config_file)),
			mac_cache(new cacheMAC(config_file)),
			bmt_cache(new cacheBMT(config_file)){}
			
		virtual void ClockTick(CPU *cpu)=0;
		virtual void getInputData(std::string data)=0;
		virtual void getInputDataType(std::string data)=0;
 		virtual int GetClk()=0;
		virtual void Cache_cycle(CPU *cpu)=0;
		//DRAM
		virtual void AccessDram(string row) =0;//flags:address  data_in_mshr:flag bit
		void ReadCallBack(uint64_t addr) { return; }
		void WriteCallBack(uint64_t addr) { return; }
		void PrintStats() { memory_system_.PrintStats(); }


		int completeTime;
		uint64_t clk_;
		string _rowDataType; //{psum,weight}
		string RorW; //{read,write}
		string general_data_row;
		string ctr_data_row;
		string bmt_data_row;
		string mac_data_row;

		MemorySystem memory_system_;
		cacheL2 *l2;
		cacheCTR *ctr_cache;
		cacheMAC *mac_cache;
		cacheBMT *bmt_cache;

	};
	class TraceBasedCPU : public CPU {
	public:
		TraceBasedCPU(const std::string& config_file, const std::string& output_dir);
		~TraceBasedCPU() {  }

		void ClockTick(CPU *cpu) override;
		void getInputData(std::string data) override;
		void getInputDataType(std::string data) override;
		int GetClk() override;
		void AccessDram(string row);

		//Cache
		void Cache_cycle(CPU *cpu);
		string Conversion_hex(unsigned long i);

		

	private:
		
 		bool get_next_;//go to next data

  		Transaction trans_;
	};
}  // namespace dramsim3
#endif

