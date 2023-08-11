#include "cache.h"
#include "cpu.h"
namespace dramsim3 {
	Cache::Cache(const std::string& config_file)
	{
		reader_cache = new INIReader(config_file);
		InitMSHR();
		delete(reader_cache);
	}
	Cache::Cache() {}

	void Cache::InitMSHR()
	{
		const auto& reader = *reader_cache;
		capacity_mshr = reader.GetInteger("L2", "capacity_mshr", 64);
		capacity_entry = reader.GetInteger("L2", "capacity_entry", 64);	//lines: 2^n
		size_mshr = 0;
		current_mshr = 0;

		//create MSHRs
		for (i = 0; i < capacity_mshr; i++)
		{
			mshr[i].item[31] = 1; //valid,MSHR is not full 
			mshr[i].item[30] = 1; //valid,MSHR is empty
		}

	}


}