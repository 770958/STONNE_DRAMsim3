#include "cacheL2.h"
#include "cpu.h"
#define GENERAL_DATA_END 1073741824		//0x4000 0000
#define GENERAL_DATA_BEGIN 134217728	//0x0800 0000

namespace dramsim3 {
	cacheL2::cacheL2(const std::string& config_file)
		:Cache(config_file)
	{
		reader_cacheL2 = new INIReader(config_file);
		InitCache();

		delete (reader_cacheL2);
	}


	void cacheL2::InitCache()
	{
		for (int i = 0; i < 6; i++) {
			this->type_num[i] = 0;
		}
		bit_block = 0;
		bit_line = 0;
		bit_set = 0;
		bit_tag = 0;
		i_cache_set = 0;
		i_num_set = 0;

		const auto& reader = *reader_cacheL2;

		this->fifo_size = reader.GetInteger("FIFO", "size", 32); //FIFO size - 1,2,4,8,16,32,64...2^18 KB

		i_cache_size = reader.GetInteger("L2", "i_cache_size", 256); //Cache size - 1,2,4,8,16,32,64...2^18 KB
		i_cache_line_size = reader.GetInteger("L2", "i_cache_line_size", 64); //Cache Line size - 1,2,4,8,16,32,64...2^18 B
		t_assoc = reader.Get("L2", "t_assoc", "SA");//mapping method - direct_mapped \ set_associative \ full_associative 


		i_cache_set = reader.GetInteger("L2", "i_cache_set", 4); //lines_num in each set- 1,2,4,8,16,32,64...2^18
		t_replace = reader.Get("L2", "replace_policy", "lru");//replace policy -  fifo\lru\lfu\random
		t_write = reader.Get("L2", "write_policy", "WB");//write policy -  write through \ write_back


		i_num_line = (i_cache_size << 10) / i_cache_line_size;

		//计算块内地址位数
		temp = i_cache_line_size;
		while (temp)
		{
			temp >>= 1;
			bit_block++;
		}
		bit_block--; //warning


		bit_line = 0; // for set_associative,the bit_line is 0
		assert(i_cache_set != 0);
		assert(i_num_line > i_cache_set);
		i_num_set = i_num_line / i_cache_set;
		temp = i_num_set;
		while (temp)
		{
			temp >>= 1;
			bit_set++;
		}
		bit_set--;

		bit_tag = 32ul - bit_block - bit_line - bit_set;
		assert(bit_tag <= 29); //32-valid-hit-dirty
		i_num_access = 0; //Number of cache access
		i_num_load = 0; //Number of cache load
		i_num_store = 0; //Number of cache store
		i_num_space = 0; //Number of space line
		i_num_hit = 0; //Number of cache hit
		i_num_load_hit = 0; //Number of load hit
		i_num_store_hit = 0; //Number of store hit
		f_ave_rate = 0.00; //Average cache hit rate
		f_load_rate = 0.00; //Cache hit rate for loads
		f_store_rate = 0.00; //Cache hit rate for stores
		current_line = 0; // The line num which is processing
		current_set = 0; // The set num which is processing
		i = 0;
		j = 0; //For loop

		//create Cache
		temp = i_num_line;
		for (i = 0; i < temp; i++)
		{
			cache_item[i][31] = true;
		}

	}

	void cacheL2::AddMiss2Entry(bitset<32> flags, int acc, dramsim3::CPU *cpu)
	{
		/*
		00: 0  Tag not match,but MSHR no space ,RANDOM select one MSHR, empty its all entries
		01: 1  Tag not match && MSHR has space, insert into a new MSHR
		10: 2  Tag match, but entry no space ,pop one
		11: 3  Tag match && entry has space, record current MSHR number
		*/
		//std::cout << "acc_num: " << acc << std::endl;
		int a = 0;
		int b = 0;
		uint64_t c = 0;
		auto pair_set = make_pair(flags, c);
		switch (acc)
		{
		case 0:
			while (capacity_mshr == size_mshr) {
				ReleaseMSHR(cpu);
			}

			//find an empty MSHR, record the num
			for (int index = 0; index < capacity_mshr; index++) {
				if (mshr[index].item[30] == 1) {
					current_mshr = index;
					break;
				}
			}
			//std::cout << "line num:  " <<current_mshr << std::endl;
			//save Tag into MSHR
			for (a = 31, b = 29; a > (31ul - bit_tag); a--, b--)
			{
				mshr[current_mshr].item[b] = flags[a];
			}

			//other bits in MSHR set 0
			for (int index = (29ul - bit_tag); index >= 0; index--)
			{
				mshr[current_mshr].item[index] = 0;
			}

			//update status
			mshr[current_mshr].item[30] = 0;//[30]MSHR not empty
			size_mshr++;

			if (this->operation == "READ")
			{

				if (cpu->act_mee == 1) {
					if (!cpu->ctr_fifo->isFull() && !cpu->mac_fifo->isFull() && !cpu->dram_fifo->isFull()) {

						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);

						this->clk_cache += cpu->mac_cache->hash_delay;

						if (this->type != "weight") {
							DataPackage* pck_ctr = new DataPackage();
							pck_ctr->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
							pck_ctr->setReq_type("READ");
							pck_ctr->setReq_source("l2");
							pck_ctr->setAddr_type(this->type);
							pck_ctr->setAddr_to_dram(this->addr);
							cpu->ctr_fifo->push(pck_ctr);
						}
						

						DataPackage* pck_mac = new DataPackage();
						pck_mac->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
						pck_mac->setReq_type("READ");
						pck_mac->setReq_source("l2");
						pck_mac->setAddr_type(this->type);
						pck_mac->setAddr_to_dram(this->addr);
						cpu->mac_fifo->push(pck_mac);
						this->get_next = true;
					}
				}
				else
				{
					if (!cpu->dram_fifo->isFull()) {
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);
						this->get_next = true;
					}
				}
				//std::cout << "dram_input" << std::endl;

				if (cpu->_rowDataType == "psum") {
					this->type_num[3]++;
				}
				else  if (cpu->_rowDataType == "weight")
				{
					this->type_num[4]++;
				}
				else
				{
					this->type_num[5]++;
				}
			}
			else
			{
				if (cpu->act_mee == 1) {
					if (!cpu->ctr_fifo->isFull() && !cpu->dram_fifo->isFull()) {
						//fetch this block
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);

						this->clk_cache += cpu->mac_cache->hash_delay;
						if (this->type != "weight") {
							DataPackage* pck_ctr = new DataPackage();
							pck_ctr->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
							pck_ctr->setReq_type("WRITE");
							pck_ctr->setReq_source("l2");
							pck_ctr->setAddr_type(this->type);
							pck_ctr->setAddr_to_dram(this->addr);
							cpu->ctr_fifo->push(pck_ctr);
						}
						
					}
				}
				else
				{
					//fetch this block
					if (!cpu->dram_fifo->isFull()) {
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);
						this->get_next = true;
					}
				}

				if (cpu->_rowDataType == "psum") {
					this->type_num[3]++;
				}
				else  if (cpu->_rowDataType == "weight")
				{
					this->type_num[4]++;
				}
				else
				{
					this->type_num[5]++;
				}
			}
			//save offset into entry
			pair_set = make_pair(flags, cpu->completeTime);
			mshr[current_mshr].entry.push(pair_set);
			break;
		case 1:
			//find an empty MSHR, record the num
			for (int index = 0; index < capacity_mshr; index++) {
				if (mshr[index].item[30] == 1) {
					current_mshr = index;
					break;
				}

			}
			//std::cout << "line num:  " <<current_mshr << std::endl;
			//save Tag into MSHR
			for (a = 31, b = 29; a > (31ul - bit_tag); a--, b--)
			{
				mshr[current_mshr].item[b] = flags[a];
			}

			//other bits in MSHR set 0
			for (int index = (29ul - bit_tag); index >= 0; index--)
			{
				mshr[current_mshr].item[index] = 0;
			}

			//update status
			mshr[current_mshr].item[30] = 0;//[30]MSHR not empty
			size_mshr++;

			if (this->operation == "READ")
			{
				if (cpu->act_mee == 1) {
					if (!cpu->ctr_fifo->isFull() && !cpu->mac_fifo->isFull() && !cpu->dram_fifo->isFull()) {

						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);

						this->clk_cache += cpu->mac_cache->hash_delay;
						if (this->type != "weight") {
							DataPackage* pck_ctr = new DataPackage();
							pck_ctr->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
							pck_ctr->setReq_type("READ");
							pck_ctr->setReq_source("l2");
							pck_ctr->setAddr_type(this->type);
							pck_ctr->setAddr_to_dram(this->addr);
							cpu->ctr_fifo->push(pck_ctr);
						}
						

						DataPackage* pck_mac = new DataPackage();
						pck_mac->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
						pck_mac->setReq_type("READ");
						pck_mac->setReq_source("l2");
						pck_mac->setAddr_type(this->type);
						pck_mac->setAddr_to_dram(this->addr);
						cpu->mac_fifo->push(pck_mac);
						this->get_next = true;
					}
				}
				else
				{
					if (!cpu->dram_fifo->isFull()) {
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);
						this->get_next = true;
					}
				}

			}
			else
			{
				if (cpu->act_mee == 1) {
					if (!cpu->ctr_fifo->isFull() && !cpu->dram_fifo->isFull()) {
						//fetch this block
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);

						this->clk_cache += cpu->mac_cache->hash_delay;
						if (this->type != "weight") {
							DataPackage* pck_ctr = new DataPackage();
							pck_ctr->setAdded_cycle(std::to_string(stoi(this->added_cycle) + this->clk_cache));
							pck_ctr->setReq_type("WRITE");
							pck_ctr->setReq_source("l2");
							pck_ctr->setAddr_type(this->type);
							pck_ctr->setAddr_to_dram(this->addr);
							cpu->ctr_fifo->push(pck_ctr);
						}
						
					}
				}
				else
				{
					//fetch this block
					if (!cpu->dram_fifo->isFull()) {
						DataPackage* pck_dram = new DataPackage();
						pck_dram->setAdded_cycle(std::to_string(std::stoi(this->added_cycle) + this->clk_cache));
						pck_dram->setReq_type("READ");
						pck_dram->setReq_source("l2");
						pck_dram->setAddr_type(this->type);
						pck_dram->setAddr_to_dram(this->addr);
						cpu->dram_fifo->push(pck_dram);
						this->get_next = true;
					}
				}
				if (cpu->_rowDataType == "psum") {
					this->type_num[3]++;
				}
				else  if (cpu->_rowDataType == "weight")
				{
					this->type_num[4]++;
				}
				else
				{
					this->type_num[5]++;
				}
			}

			//save offset into entry
			pair_set = make_pair(flags, cpu->completeTime);
			mshr[current_mshr].entry.push(pair_set);

			break;
		case 2:
			while (mshr[current_mshr].entry.size() == capacity_entry)
			{
				ReleaseMSHR(cpu);
			}
			pair_set = make_pair(flags, cpu->completeTime);
			mshr[current_mshr].entry.push(pair_set);

			break;
		case 3:
			//save  into entry

			pair_set = make_pair(flags, cpu->completeTime);
			mshr[current_mshr].entry.push(pair_set);

			//update status
			if (mshr[current_mshr].entry.size() == capacity_entry) {
				mshr[current_mshr].item[31] = 0;	//[31]MSHR invalid
			}

			break;
		}
	}

	void cacheL2::GetWrite(CPU *cpu) {
		//write back to dram
		if (cpu->act_mee == 1) {
			this->clk_cache += cpu->ctr_cache->encry_delay;
		}

		if (cpu->_rowDataType == "psum") {
			this->type_num[3]++;
		}
		else  if (cpu->_rowDataType == "weight")
		{
			this->type_num[4]++;
		}
		else
		{
			this->type_num[5]++;
		}

		cache_item[current_line][29] = false; // dirty bit - false
		cache_item[current_line][30] = false; //hit bit - false
	}

	void cacheL2::cache_clock(dramsim3::CPU *cpu)
	{
		if (cpu->act_cache == 1) {
			if (!cpu->l2_fifo_rtn->isEmpty()) {
				cpu->npu_fifo_rtn->push(cpu->l2_fifo_rtn->front());
				cpu->l2_fifo_rtn->pop();
			}
		}
 		if (!cpu->l2_fifo->isEmpty()) {

			if (this->get_next) {
				/*if (cpu->isFinished()) {
					std::cout << "l2 size front: " << cpu->l2_fifo->size() << std::endl;
				}*/
				this->get_next = false;
				char *op_addr;
				unsigned long num = 0;
				op_addr = new char[cpu->l2_fifo->front()->getAddr_to_dram().length() + 1]; // 分配足够的内存空间，包括终止符'\0'
				cpu->l2_fifo->front()->getAddr_to_dram().copy(op_addr, cpu->l2_fifo->front()->getAddr_to_dram().length());
				op_addr[cpu->l2_fifo->front()->getAddr_to_dram().length()] = '\0'; // 手动添加终止符'\0'
				bitset<32> bit_req_data(strtoul(op_addr, NULL, 16));
				//*****general data*****//
				num = bit_req_data.to_ulong() % (GENERAL_DATA_END - GENERAL_DATA_BEGIN + 1) + GENERAL_DATA_BEGIN;
				cpu->general_data_row = cpu->Conversion_hex(num);
				cpu->l2_fifo->front()->setAddr_to_dram(cpu->general_data_row);
				//std::cout << "l2--2: " << cpu->l2_fifo->front()->getAddr_to_dram() << std::endl;

				cpu->general_data_row += " " + cpu->l2_fifo->front()->getReq_type() + " " + cpu->l2_fifo->front()->getAdded_cycle();

				cpu->_rowDataType = cpu->l2_fifo->front()->getAddr_type();
				cpu->RorW = (cpu->general_data_row.substr(11, 4) == "READ") ? "l" : "s";

				string op_type;
				/*l2*/
				op_type = (cpu->general_data_row.substr(11, 4) == "READ") ? "l" : "s";
				op_type += " ";
				op_type += cpu->general_data_row.substr(0, 10).c_str();
				address = new char[op_type.length() + 1]; // 分配足够的内存空间，包括终止符'\0'
				op_type.copy(address, op_type.length());
				address[op_type.length()] = '\0'; // 手动添加终止符'\0'

				this->addr = cpu->l2_fifo->front()->getAddr_to_dram();
				this->operation = cpu->l2_fifo->front()->getReq_type();
				this->added_cycle = cpu->l2_fifo->front()->getAdded_cycle();
				this->type = cpu->l2_fifo->front()->getAddr_type();

				DataPackage* del_pck_ptr = cpu->l2_fifo->front();
				cpu->l2_fifo->pop();
				delete del_pck_ptr;
				/*if (cpu->isFinished()) {
					std::cout << "l2 size  back: " << cpu->l2_fifo->size() << std::endl;
				}*/
			}
			if (std::stoi(this->added_cycle) <= this->clk_cache) {

				bool is_store = false;
				bool is_load = false;
				bool is_space = false;
				bool hit = false;
				//std::cout << "cpu：：GetHitNUm" << std::endl;
				switch (address[0])
				{
				case 's':
					is_store = true;
					break;

				case 'l':
					cpu->total_read_num++;
					is_load = true;
					break;

					//case ' ':break; //Waring if a line has nothing,the first of it is a '\0' nor a ' '
				case '\0':
					is_space = true;
					break; //In case of space lines

				default:
					cout << "L2  The address[0] is:" << address[0] << endl;
					cout << "ERROR IN JUDGE!" << endl;
				}
				temp = strtoul(address + 2, NULL, 16);
				bitset<32> flags(temp); // flags if the binary of address
				hit = IsHit(flags);


				// hit，load
				if (hit && is_load)
				{
					if (cpu->_rowDataType == "psum") {
						this->type_num[0]++;
					}
					else if (cpu->_rowDataType == "weight")
					{
						this->type_num[1]++;
					}
					else
					{
						this->type_num[2]++;
					}
					i_num_access++;
					i_num_load++;
					i_num_load_hit++;
					i_num_hit++;
					if (t_replace == "lru")
					{
						LruHitProcess();
					}

					cpu->l2_fifo_rtn->push(this->clk_cache - this->clk_last_cache);
					this->clk_last_cache = this->clk_cache;
					this->get_next = true;


				}

				// hit，store
				else if (hit && is_store)
				{
					if (cpu->_rowDataType == "psum") {
						this->type_num[0]++;
					}
					else if (cpu->_rowDataType == "weight")
					{
						this->type_num[1]++;
					}
					else
					{
						this->type_num[2]++;
					}
					i_num_access++;
					i_num_store++;
					i_num_store_hit++;
					i_num_hit++;
					cache_item[current_line][29] = true; //设置dirty为true
					if (t_replace == "lru")
					{
						LruHitProcess();
					}
					this->get_next = true;

				}
				// not hit，load
				else if ((!hit) && is_load)
				{
					i_num_access++;
					i_num_load++;
					MSHR_cycle(flags, cpu);//mshr
					//std::cout << "read_miss "  << std::endl;
					GetRead(flags, cpu); // read data from memory
					if (t_replace == "lru")
					{
						LruUnhitSpace();
					}
				}
				// not hit, store
				else if ((!hit) && is_store)
				{
					i_num_access++;
					i_num_store++;
					MSHR_cycle(flags, cpu);//mshr
					//std::cout << "write_miss " << std::endl;

					GetRead(flags, cpu); // read data from memory
					cache_item[current_line][29] = true; //set dirty bit to true
					if (t_replace == "lru")
					{
						LruUnhitSpace();
					}
				}
				//no data from trace
				else if (is_space)
				{
					i_num_space++;
				}
				else
				{
					cerr << "Something ERROR" << endl;
				}

				//std::cout<< "fifo_end_1" << std::endl;
			}
		}
		this->clk_cache++;



	}

	bool cacheL2::IsHit(bitset<32> flags)
	{
		bool ret = false;


		bitset<32> flags_set;

		for (j = 0, i = (bit_block); i < (bit_block + bit_set); j++, i++) //find set num 
		{
			flags_set[j] = flags[i];
		}

		current_set = flags_set.to_ulong();

		//N-ways SA
		for (temp = (current_set*i_cache_set); temp < ((current_set + 1)*i_cache_set); temp++)
		{
			if (cache_item[temp][30] == true) //can hit or not,judge from hit bit
			{
				ret = true;

				for (i = 31, j = 28; i > (31ul - bit_tag); i--, j--) //the Tag can match or not, i:address,j:cache
				{
					if (flags[i] != cache_item[temp][j])
					{
						ret = false;
						break;
					}
				}
			}

			if (ret == true)
			{
				current_line = temp;
				break;
			}
		}

		//	ret = false;	//no cache module

		return ret;
	}

	void cacheL2::LruHitProcess() // if the replacement policy is LRU,and hit
	{

		for (i = (current_set*i_cache_set); i < ((current_set + 1)*i_cache_set); i++)
		{
			if (LRU_priority[i] < LRU_priority[current_line] && cache_item[current_line][30] == true)
			{
				LRU_priority[i]++; //if the i num line is smaller than current line,and hit bit is true 
			}
		}

		LRU_priority[current_line] = 0;

	}

	void cacheL2::LruUnhitSpace() // if the replacement policy is LRU,and not hit,but there has a spaceline
	{
		for (i = (current_set*i_cache_set); i < ((current_set + 1)*i_cache_set); i++)
		{
			if (cache_item[current_line][30] == true)
			{
				LRU_priority[i]++;
			}
		}

		LRU_priority[current_line] = 0;

	}

	void cacheL2::LruUnhitUnspace()
	{
		temp = LRU_priority[current_set*i_cache_set];

		for (i = (current_set*i_cache_set); i < ((current_set + 1)*i_cache_set); i++)
		{
			if (LRU_priority[i] >= temp)
			{
				temp = LRU_priority[i];
				j = i;
			}
		}

		current_line = j;

	}

	void cacheL2::GetRead(bitset<32> flags, CPU *cpu)
	{
		bool space = false;

		for (temp = (current_set*i_cache_set); temp < ((current_set + 1)*i_cache_set); temp++)
		{
			if (cache_item[temp][30] == false) //find a space line
			{
				space = true;
				break;
			}
		}

		if (space == true)
		{
			current_line = temp;

			for (i = 31, j = 28; i > (31ul - bit_tag); i--, j--)
			{
				cache_item[current_line][j] = flags[i];
				assert(j > 0);
			}

			cache_item[current_line][30] = true;

			if (t_replace == "lru")
			{
				LruUnhitSpace();
			}
		}
		else
		{
			GetReplace(flags, cpu);
		}

	}

	void cacheL2::GetReplace(bitset<32> flags, CPU *cpu)
	{
		if (t_replace == "random")
		{
			temp = rand() / (RAND_MAX / i_cache_set + 1); // a random line in(0,i_cache_set-1)
			current_line = current_set * i_cache_set + temp; // a random line in current_set
		}
		else if (t_replace == "lru")
		{
			LruUnhitUnspace();
		}


		if (cache_item[current_line][29] == true) //write to dram only if the dirty bit is true 
		{
			GetWrite(cpu);

		}

		for (i = 31, j = 28; i > (31ul - bit_tag); i--, j--) //set Tag bits
		{
			cache_item[current_line][j] = flags[i];
			assert(j > 0);
		}

		cache_item[current_line][30] = true; //set hit bit to true
	}

	std::pair<bool, bool> cacheL2::WillAcceptMiss(bitset<32> flags)
	{
		/*
			00:Tag not match && MSHR no space, wait for MSHR
			01:Tag not match, but MSHR has space, insert into a new MSHR
			10:Tag match, but entry no space, wait for entry
			11:Tag match && entry has space, record current MSHR number
		*/
		int a = 0;
		int b = 0;
		std::pair<bool, bool> acc;
		acc = make_pair(1, 0);
		//match miss Tag in the whole MSHRs,	index: MSHR num 
		for (int index = 0; index < capacity_mshr; index++)
		{
			//std::cout << "items: " << mshr[index].item << std::endl;

			//invalid MSHR,full 
			if (mshr[index].item[31] == 0) {
				continue;
			}
			//valid MSHR,not full
			else
			{
				//the Tag can match or not, a:address,b:MSHR
				for (a = 31, b = 29; a > (31ul - bit_tag); a--, b--)
				{
					if (flags[a] == mshr[index].item[b])
					{
						continue;
					}
					else
					{
						acc.first = false;
						break;
					}

				}
				// Tag match 
				if (acc.first == true)
				{
					current_mshr = index;
					// entry has space
					if (mshr[index].entry.size() < capacity_entry) {
						acc.second = 1;
					}
					// entry has no space
					else
					{
						acc.second = 0;
					}
					return acc;
				}
			}

		}
		// Tag not match && MSHR has space,  insert into a new MSHR
		if (!acc.first) {
			//MSHR has space
			if (size_mshr < capacity_mshr) {
				acc.second = 1;
			}
			//MSHR has no space
			else
			{
				acc.second = 0;
			}
		}
		return acc;
	}

	void cacheL2::ReleaseMSHR(dramsim3::CPU *cpu)
	{
		for (int index = 0; index < capacity_mshr; index++)
		{
			while ((!mshr[index].entry.empty()) && mshr[index].entry.front().second < cpu->clk_) {
				mshr[index].entry.pop();
			}
			if (mshr[index].entry.empty()) {
				mshr[index].item[31] = 1;
				mshr[index].item[30] = 1;
				size_mshr--;
			}
		}
	}

	void cacheL2::MSHR_cycle(bitset<32> flags, dramsim3::CPU *cpu)
	{
		int MSHR_type = 0;
		//std::cout << "flags:   " << flags << std::endl;
		//std::cout << "enter cycle" << std::endl;
		std::pair<bool, bool> acc = WillAcceptMiss(flags);
		if (acc.first == 0 && acc.second == 0) {
			MSHR_type = 0;
		}
		else if (acc.first == 0 && acc.second == 1) {
			MSHR_type = 1;
		}
		else if (acc.first == 1 && acc.second == 0) {
			MSHR_type = 2;
		}
		else {
			MSHR_type = 3;
		}

		//	MSHR_type = 0;	//no MSHR module

		AddMiss2Entry(flags, MSHR_type, cpu);
		ReleaseMSHR(cpu);
		//std::cout << bit_tag << std::endl;

	}

	void cacheL2::PrintCache(string layer_name, string index_layer_, string file)
	{

		string l2_out_file = file + "/l2cache.csv";
		ofstream ofs;
		ofs.open(l2_out_file, ios::out | ios::app);

		f_ave_rate = ((double)i_num_hit) / i_num_access; //Average cache hit rate
		f_load_rate = ((double)i_num_load_hit) / i_num_load; //Cache hit rate for loads
		f_store_rate = ((double)i_num_store_hit) / i_num_store; //Cache hit rate for stores

		if (index_layer_ == "0001") {
			ofs << "layer_name" << "," << "cache_access_total_num" << "," << "cache_access_load_num" << "," << "cache_access_store_num" << ","
				<< "cache_hit_load_num" << "," << "cache_hit_store_num" << ","
				<< "psum_cache_access" << "," << "psum_dram_access" << ","
				<< "weight_cache_access" << "," << "weight_dram_access" << ","
				<< "feature_map_cache_access" << "," << "feature_map_dram_access" << ","
				<< "cache_hit_rate" << "," << endl;
		}

		ofs << layer_name + index_layer_ << "," << i_num_access << "," << i_num_load << "," << i_num_store << ","
			<< i_num_load_hit << "," << i_num_store_hit << ","
			<< type_num[0] << "," << type_num[3] << ","
			<< type_num[1] << "," << type_num[4] << ","
			<< type_num[2] << "," << type_num[5] << ","
			<< f_ave_rate << "," << endl;


		ofs.close();
		//std::cout << "Printing cache end" << std::endl;

	}
	/*
	void cacheL2::PrintCache(string layer_name, string index_layer_)
	{


		string l2_out_file = "../dramsim3/output/l2cache.csv";
		std::cout << "Printing L2 cache" << std::endl;
		ofstream ofs;
		ofs.open(l2_out_file, ios::out|ios::app);

		f_ave_rate = ((double)i_num_hit) / i_num_access; //Average cache hit rate
		f_load_rate = ((double)i_num_load_hit) / i_num_load; //Cache hit rate for loads
		f_store_rate = ((double)i_num_store_hit) / i_num_store; //Cache hit rate for stores

		if (index_layer_=="0001") {
			ofs << "layer_name" << "," << "cache_access_total_num" << "," << "cache_access_load_num" << "," << "cache_access_store_num" << ","
				<< "cache_hit_load_num" << "," << "cache_hit_store_num" << ","
				<< "psum_cache_access" << "," << "psum_dram_access" << ","
				<< "weight_cache_access" << "," << "weight_dram_access" << ","
				<< "feature_map_cache_access" << "," << "feature_map_dram_access" << ","
				<< "cache_hit_rate" << "," << endl;
		}

		ofs << layer_name + index_layer_ << "," << i_num_access << "," << i_num_load << "," << i_num_store << ","
			<< i_num_load_hit << "," << i_num_store_hit << ","
			<< type_num[0] << "," << type_num[3] << ","
			<< type_num[1] << "," << type_num[4] << ","
			<< type_num[2] << "," << type_num[5] << ","
			<< f_ave_rate << "," << endl;


		ofs.close();
		std::cout << "Printing cache end" << std::endl;

	}
	*/
}