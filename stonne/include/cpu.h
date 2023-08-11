//
// Created by wangjs
//
#ifndef __CPU_H
#define __CPU_H

#include <fstream>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <bitset>
#include <vector>
#include "memory_system.h"
#include "cacheL2.h"
#include "cacheCTR.h"
#include "cacheBMT.h"
#include "cacheMAC.h"
#include "../ext/headers/INIReader.h"
#include "cacheFifo.h"
#include "Fifo.h"
#include <type_traits>
using namespace std;
 namespace dramsim3 {

	class CPU {
	public:

		CPU(const std::string& config_file, const std::string& output_dir)
			: memory_system_(
				config_file, output_dir,
				std::bind(&CPU::ReadCallBack, this, std::placeholders::_1),
				std::bind(&CPU::WriteCallBack, this, std::placeholders::_1)),
			address_general{"l 0x00000000"}, address_ctr{ "l 0x00000000" },
			address_bmt{ "l 0x00000000" }, address_mac{ "l 0x00000000" },
			clk_(0),
			current_level(0),
			completeTime(0),
			ctr_block_num(0),
			l2(new cacheL2(config_file)), 
			ctr_cache(new cacheCTR(config_file)),mac_cache(new cacheMAC(config_file)),bmt_cache(new cacheBMT(config_file)),
			bmt_fifo(new Fifo(32 * 1024 * 8)),mac_fifo(new Fifo(32 * 1024 * 8)),ctr_fifo(new Fifo(32 * 1024 * 8)),
			l2_fifo(new Fifo(32 * 1024 * 8)), dram_fifo(new Fifo(32 * 1024 * 8)),
			l2_fifo_rtn(new cacheFifo(32 * 1024 * 8)), npu_fifo_rtn(new cacheFifo(32 * 1024 * 8)){}
			
		//int fifo_size = 0;
		virtual void setAccessConfig()=0;
		//virtual void getInputData(std::string data)=0;
		//virtual void getInputDataType(std::string data)=0;
		//virtual int GetClk() = 0;
		virtual string Conversion_hex(unsigned long long i)=0;
		//virtual int SetClk() = 0;
		virtual void setTracefile(std::string s) = 0;//flags:address  data_in_mshr:flag bit
		virtual bool isFinished() = 0;//flags:address  data_in_mshr:flag bit

 		//DRAM
		virtual void AccessDram( ) =0;//flags:address  data_in_mshr:flag bit
		void ReadCallBack(uint64_t addr) { return; }
		void WriteCallBack(uint64_t addr) { return; }
		void PrintStats() { memory_system_.PrintStats(); }
		virtual void GetDelay() = 0;
		char address_general[13]; //address to L2, e.g. "l 0xffffffff"
		char address_ctr[13];
		//char address_bmt[13];
		vector<char*>address_bmt;
		char address_mac[13];

		int mee_delay = 0;
		int mee_last_delay = 0;
		int completeTime;
		int lastcomptime;
		uint64_t clk_;
		string _rowDataType; //{psum,weight,feature_map}
		string RorW; //{read,write}
		string general_data_row;
		string ctr_data_row;
		vector<string>bmt_data_row;//7G-General --> 112MB-CTR --> 7级16叉BMT树,root存储在系统内部
  
		unsigned int current_level;
		string mac_data_row;
		unsigned long ctr_block_num = 0;

		MemorySystem memory_system_;

		cacheL2 *l2;
		cacheCTR *ctr_cache;
		cacheMAC *mac_cache;
		cacheBMT *bmt_cache;

		Fifo* mac_fifo;
		Fifo* ctr_fifo;
		Fifo* bmt_fifo;
		Fifo* l2_fifo;
		Fifo* dram_fifo;

		std::queue<std::string> req_fifo;

		cacheFifo* l2_fifo_rtn; 
		cacheFifo* npu_fifo_rtn;

		std::string addr_genel;
		std::string addr_mac;
		std::vector<std::string> addr_bmt;
		std::string addr_ctr;

 		int act_mee;
		int act_cache;
		int mac_length;
		int bmt_length;
		int block_length;

		unsigned long long total_read_delay = 0;
		double avg_read_delay = 0.00;
		unsigned long long total_read_num = 0;
		unsigned long long partition_cycle = 0;
		unsigned long long n_cycles = 0;
		int cfg_num = 0;
		int npu_num = 0;
		
		bool tpu_filter = false;

		std::string line;
		std::string model_name; 
		std::string arch_name;
		std::string dram_name;
		
		//float test_layer_num = 0.0;

	};
	class TraceBasedCPU : public CPU {
	public:
		TraceBasedCPU(const std::string& config_file, const std::string& output_dir);
		~TraceBasedCPU() 
		{		}

		void setAccessConfig() override;
		void getInputData(std::string data);
		//void getInputDataType(std::string data) override;
		//int GetClk() override;
		void AccessDram( )override;
		void setTracefile(std::string s)override;
		//Cache
		//void Cache_cycle(CPU *cpu);
		//void No_cache_cycle(CPU *cpu); 
		string Conversion_hex(unsigned long long i)override;
		bool isFinished()override;
		void GetDelay()override;
	private:
		string outfile;
		unsigned int clk_set = 0;
		std::string address, operation, added_cycle, type;
		std::queue<int> read_delay;
		std::ifstream trace_file_;
		string trace_file;
		bool get_next_;//go to next data
		bool get_next_trace;//go to next data
		bool l2_push_finished = false;//go to next data
		bool mac_push_finished = false;//go to next data
		bool ctr_push_finished = false;//go to next data
		bool bmt_push_finished_ = false;//go to next data
		bool bmt_push_finished[6] = { false, false, false, false, false, false };
		INIReader* reader_act;
  		Transaction trans_;
		bool cpu_finished = false;
		int comp = 0;

	};
}  // namespace dramsim3
#endif

