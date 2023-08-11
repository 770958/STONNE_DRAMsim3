#ifndef _CACHE_H_
#define _CACHE_H_

#include <fstream>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <queue>

#include <bitset>
#include "../ext/headers/INIReader.h"
#include "cacheFifo.h"
#define MAX_CACHE_LINE 65536 // 65536(2^16)

using namespace std;


namespace dramsim3 {
	struct MSHR {
		std::queue<std::pair<std::bitset<32>, uint64_t>>entry;// addr,thisCompleteTime
		std::bitset<32> item = { 0 };//[31]:MSHR valid,[30]:empty  
	};
	class CPU;	//前向声明

	class Cache {
	public:
		Cache(const std::string& config_file);
		Cache();
		~Cache() {}

		//L2 Cache 
	/******************************************/
		unsigned int long i_cache_size; //cache size
		unsigned int long i_cache_line_size; //cacheline size
		unsigned int long i_cache_set; //cache set

		unsigned int long i_num_line; //How many lines of the cache.
		unsigned int long i_num_set; //How many sets of the cache.

		std::string t_assoc; //associativity method
		std::string t_replace; //replacement policy
		std::string t_write; //write policy
	/******************************************/

	/******************************************/
		short unsigned int bit_block; //How many bits of the block. sign cache line size
		short unsigned int bit_line; //How many bits of the line.
		short unsigned int bit_tag; //How many bits of the tag.	
		short unsigned int bit_set; //How many bits of the set.	sign set numbers
	/******************************************/

	/******************************************/
		unsigned long int i_num_access; //Number of cache access
		unsigned long int i_num_load; //Number of cache load
		unsigned long int i_num_store; //Number of cache store
		unsigned long int i_num_space; //Number of space line

		unsigned long int i_num_hit; //Number of cache hit
		unsigned long int i_num_load_hit; //Number of load hit
		unsigned long int i_num_store_hit; //Number of store hit

		double f_ave_rate; //Average cache hit rate
		double f_load_rate; //Cache hit rate for loads
		double f_store_rate; //Cache hit rate for stores
	/******************************************/

		std::bitset<32> cache_item[MAX_CACHE_LINE] = { 0 }; // [31]:valid,[30]:hit,[29]:dirty,[28]-[0]:data
		unsigned long int LRU_priority[MAX_CACHE_LINE] = { 0 }; //For LRU policy's priority
		unsigned long int current_line; // The line num which is processing
		unsigned long int current_set; // The set num which is processing
		unsigned long int i, j; //For loop
		unsigned long int temp; //A temp varibale
	/******************************************/
		unsigned int clk_cache = 0;
		unsigned int clk_last_cache = 0;
		virtual void InitCache() = 0;

		// Cache
		virtual void cache_clock(dramsim3::CPU *cpu) = 0;
		virtual bool IsHit(bitset<32> flags) = 0;
		virtual void LruHitProcess() = 0;
		virtual void LruUnhitSpace() = 0;
		virtual void LruUnhitUnspace() = 0;
		virtual void GetRead(bitset<32> flags, CPU *cpu) = 0;
		virtual void GetReplace(bitset<32> flags, CPU *cpu) = 0;
		virtual void GetWrite(CPU *cpu) = 0; //写入内存
		virtual void PrintCache(string layer_name, string index_layer_, string file) = 0;

		INIReader* reader_cache;
		unsigned long type_num[6] = { 0 }; //data access type from npu ----"cache[psum],cache[weight],cache[feature_map],dram[psum],dram[weight],dram[feature_map]"
		/*MSHR*/
	/******************************************/
		unsigned int capacity_mshr;
		unsigned int capacity_entry;
		unsigned int size_mshr;
		unsigned int current_mshr;// The MSHR num which is processing
	/******************************************/
		virtual void MSHR_cycle(bitset<32> flags, dramsim3::CPU *cpu) = 0;
		virtual std::pair<bool, bool> WillAcceptMiss(bitset<32> flags) = 0;
		virtual void AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu) = 0;
		virtual void ReleaseMSHR(dramsim3::CPU *cpu) = 0;
		/******************************************/
		MSHR mshr[MAX_CACHE_LINE];
		void InitMSHR();

		/******************************************/
	};
}
#endif
