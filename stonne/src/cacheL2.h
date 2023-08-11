#ifndef _CACHEL2_H_
#define _CACHEL2_H_

#include"cache.h"

namespace dramsim3 {

	//L2
	class cacheL2 :public Cache {
	public:
		INIReader* reader_cacheL2;

		cacheL2(const std::string& config_file);
		void InitCache()override;
		void GetHitNum(char *address, dramsim3::CPU *cpu)override;
		void PrintCache(std::string file_name)override;
	};
}


#endif