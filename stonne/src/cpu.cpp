// created by wangjs, modified according to dramsim3

#include "cpu.h"
#include"cache.h"
#include <iomanip>
#define GENERAL_DATA_END 1073741824		//0x4000 0000
#define GENERAL_DATA_BEGIN 134217728	//0x0800 0000
#define MAC_DATA_BEGIN 16777216			//0x0100 0000
#define CTR_DATA_BEGIN 2097152			//0x0020 0000
namespace dramsim3 {

	TraceBasedCPU::TraceBasedCPU(const std::string & config_file, 
		const std::string & output_dir) : CPU(config_file, output_dir)
	{
		this->completeTime = 0;
		this->lastcomptime = 0;
		this->bmt_length = bmt_cache->bmt_length;
		this->mac_length = mac_cache->mac_length;
		this->block_length = this->l2->i_cache_line_size;
		this->outfile = output_dir;
 		get_next_ = true; 
		get_next_trace = true;
		reader_act = new INIReader(config_file);
		const auto& reader = *reader_act;
		act_cache = reader.GetInteger("act_config", "act_cache", 1); 
		act_mee = reader.GetInteger("act_config", "act_mee", 1); 
  		//std::cout << "TraceBasedCPU cre end" << std::endl;
		delete(reader_act); 
	}

	void TraceBasedCPU::setAccessConfig()
	{ 
		if (!this->req_fifo.empty()) { 
			if (this->get_next_trace) {
				this->get_next_trace = false;
				//std::getline(trace_file_, line);
				this->line = this->req_fifo.front();
				this->req_fifo.pop();
				std::istringstream iss(line); 
				iss >> address >> operation >> added_cycle >> type;
				this->cfg_num++;
				
			}
			if (!added_cycle.empty() && std::stoi(added_cycle) <= clk_set) {

				if (this->act_cache == 1) 
				{
					if (!this->l2_fifo->isFull()) {
						DataPackage* pck_l2 = new DataPackage();
						pck_l2->setAdded_cycle(added_cycle);
						pck_l2->setReq_type(operation);
						pck_l2->setAddr_to_dram(address);
						pck_l2->setAddr_type(type);
						pck_l2->setReq_source("npu");
						this->l2_fifo->push(pck_l2);
						this->get_next_trace = true;
					}
				}
				else
				{
					this->getInputData(address);
					if (this->act_mee == 1) {
						if (!this->dram_fifo->isFull() && !this->l2_push_finished) {
							DataPackage* pck_dram = new DataPackage();
							pck_dram->setAdded_cycle(added_cycle);
							pck_dram->setReq_type(operation);
							pck_dram->setAddr_type(type);
							pck_dram->setAddr_to_dram(this->addr_genel);
							pck_dram->setReq_source("l2");
							this->dram_fifo->push(pck_dram);
							if (operation == "READ") {
								this->total_read_num++;
							}
							this->l2_push_finished = true;
						}

						if (!this->dram_fifo->isFull() && !this->mac_push_finished) {
							DataPackage* pck_mac = new DataPackage();
							pck_mac->setAdded_cycle(added_cycle);
							pck_mac->setReq_type(operation);
							pck_mac->setAddr_type(type);
							pck_mac->setAddr_to_dram(this->addr_mac);
							pck_mac->setReq_source("mac");
							this->dram_fifo->push(pck_mac);
							if (operation == "READ") {
								this->total_read_num++;
							}
							this->mac_push_finished = true;
						}

						if (!this->dram_fifo->isFull() && !this->ctr_push_finished ) {
							
							DataPackage* pck_ctr = new DataPackage();
							pck_ctr->setAdded_cycle(added_cycle);
							pck_ctr->setReq_type(operation);
							pck_ctr->setAddr_type(type);
							pck_ctr->setAddr_to_dram(this->addr_ctr);
							pck_ctr->setReq_source("ctr");
							this->dram_fifo->push(pck_ctr);
							if (operation == "READ") {
								this->total_read_num++;
							}
							
							this->ctr_push_finished = true;
						}
						

						for (int i = 0; i < 6; i++) {
							if (!this->dram_fifo->isFull() && !this->bmt_push_finished[i]) {
								DataPackage* pck_bmt = new DataPackage();
								std::string s = this->addr_bmt[i];
								pck_bmt->setAdded_cycle(added_cycle);
								pck_bmt->setReq_type(operation);
								pck_bmt->setAddr_type(type);
								pck_bmt->setAddr_to_dram(s);
								pck_bmt->setReq_source("bmt");
								this->dram_fifo->push(pck_bmt);
								if (operation == "READ") {
									this->total_read_num++;
								}
								this->bmt_push_finished[i] = true;
							}
						}

						if (this->bmt_push_finished[5]) {
							this->bmt_push_finished_ = true;
							for (int j = 0; j < 6; j++) {
								this->bmt_push_finished[j] = false;
							}
						}

						if (this->l2_push_finished&&this->mac_push_finished&&this->ctr_push_finished&&this->bmt_push_finished_) { 
							while (!this->addr_bmt.empty())
							{
								this->addr_bmt.erase(this->addr_bmt.begin());
							}
							this->l2_push_finished = false;
							this->mac_push_finished = false;
							this->ctr_push_finished = false;

							this->get_next_trace = true;
						}
					}
					else
					{
						if (!this->dram_fifo->isFull()) {
							DataPackage* pck_dram = new DataPackage();
							pck_dram->setAdded_cycle(added_cycle);
							pck_dram->setReq_type(operation);
							pck_dram->setAddr_type(type);
							pck_dram->setAddr_to_dram(this->addr_genel);
							pck_dram->setReq_source("l2");
							this->dram_fifo->push(pck_dram);
							if (operation == "READ") {
								this->total_read_num++;
							}
							this->get_next_trace = true;
						}
					}
				}
			}
			
		}
		

		if ((this->npu_num > 0) && this->req_fifo.empty() && (this->npu_num==this->cfg_num))
		{
			//std::cout << "config request done" << std::endl;
			//std::cout << this->npu_num << "  "<< this->cfg_num << std::endl;

			cpu_finished = true;
		}

		clk_set++;
  	}

	void TraceBasedCPU::AccessDram()
	{
		//std::cout << "ori: " << this->dram_fifo->size() << std::endl;

		if (this->act_cache == 0) {
			if (!this->l2_fifo_rtn->isEmpty()) {
 				this->npu_fifo_rtn->push(this->l2_fifo_rtn->front());
				this->l2_fifo_rtn->pop();
			}
		} 
		if (!this->dram_fifo->isEmpty()) {

			if (this->dram_fifo->front()->getAddr_type() == "weight" && this->dram_fifo->front()->getReq_source() == "ctr" && this->dram_fifo->front()->getReq_type() != "READ") {
				DataPackage* del_pck = this->dram_fifo->pop();
				delete del_pck;
			}
			if (!this->dram_fifo->isEmpty()) 
			{
				std::queue<int> get_delay_queue;
				//std::cout << "dram_read" << std::endl;
				memory_system_.ClockTick(); 
			
				//until the final request
				if (this->get_next_)
				{
					this->get_next_ = false;

					string row = this->dram_fifo->front()->getAddr_to_dram() + " " + this->dram_fifo->front()->getReq_type() + " " + this->dram_fifo->front()->getAdded_cycle();
					std::stringstream rowData(row);
					rowData >> trans_;
					trans_.addr_str = row.substr(0, 10);

					DataPackage* del_pck = this->dram_fifo->pop();
					delete del_pck;
				}

				if (trans_.added_cycle <= clk_)
				{
					this->get_next_ = memory_system_.WillAcceptTransaction(trans_.addr,
						trans_.is_write);
					if (this->get_next_)
					{
						memory_system_.AddTransaction(trans_.addr, trans_.is_write, trans_.addr_str);
						clk_++;
					}

				}

				while (trans_.added_cycle > clk_) {
					clk_++;
					memory_system_.ClockTick();
				}
				get_delay_queue = memory_system_.ReturnDelay();
				while (!get_delay_queue.empty())
				{
					this->l2_fifo_rtn->push(get_delay_queue.front());
					get_delay_queue.pop();
				}
				clk_++;
				this->completeTime = clk_;
				//this->lastcomptime = this->completeTime;

				//std::cout << "this->dram_fifo->pop(): " << std::endl;

			}
			
		}
		//std::cout << "now: " << this->dram_fifo->size() << std::endl;

	}

	void TraceBasedCPU::setTracefile(std::string s)
	{
		trace_file = s;
		trace_file_.open(trace_file);
		if (trace_file_.fail()) {
			std::cerr << "Trace file does not exist" << std::endl;
			AbruptExit(__FILE__, __LINE__);
		}
	}

	bool TraceBasedCPU::isFinished( )
	{
		return cpu_finished;
	}

	void TraceBasedCPU::GetDelay()
	{
 		//this->total_read_delay = 0;
		unsigned int remained = this->npu_fifo_rtn->size();;
		unsigned int progressCount = remained / 100; // 计算每份的数量
		unsigned int progress = 0;  // 记录进度 

 		while (!this->npu_fifo_rtn->isEmpty()) {
			if (this->npu_fifo_rtn->front() >= 0) {
				this->total_read_delay += this->npu_fifo_rtn->front();
				this->npu_fifo_rtn->pop();
 			}
			else
			{
 				this->npu_fifo_rtn->pop();
			}
			remained--;

			if (remained % progressCount == 0)
			{
				progress++;
				if (progress == 100) {
					std::cout << "Finished: " << this->model_name << " " << this->arch_name << " " << this->dram_name << std::endl;
				}
			}
 		}
	}

	//L2 Cache  
	string TraceBasedCPU::Conversion_hex(unsigned long long i) //将int转成16进制字符串
	{
		stringstream ioss; //定义字符串流
		string s_temp; //存放转化后字符
		ioss << setiosflags(ios::uppercase) << hex << i; //以十六制(大写)形式输出
		//ioss << resetiosflags(ios::uppercase) << hex << i; //以十六制(小写)形式输出//取消大写的设置
		ioss >> s_temp;
		if (s_temp.size() > 8) {
			s_temp = s_temp.substr(0, 8);
		}
		while (s_temp.length() < 8) {
			s_temp = '0' + s_temp;
		}
		s_temp = "0x" + s_temp;
		return s_temp;
	}

	void TraceBasedCPU::getInputData(string data)
	{

		//32位访存地址请求，若先不考虑rank、bank，在整个二维平面观察DRAM
		//地址会在DRAMsim内部重映射
		//0-13：列地址  14-31：行地址

		char op_addr[11];
		unsigned long num_addr = 0;
		strcpy(op_addr, (data.substr(0, 10)).c_str());
		bitset<32> bit_req_data(strtoul(op_addr, NULL, 16));
		//*****general data*****//
		num_addr = bit_req_data.to_ulong() % (GENERAL_DATA_END - GENERAL_DATA_BEGIN + 1) + GENERAL_DATA_BEGIN;
		addr_genel = Conversion_hex(num_addr);
		//*****MAC*****//
		bitset<32> mac_flag(0);
		for (int m = 0, n = 0; m < 14; m++, n++) {
			mac_flag[n] = bit_req_data[m];
		}
		num_addr = mac_flag.to_ulong() % (64 / 8);
		bitset<14> mac_offset(num_addr);

		mac_flag.reset();
		for (int m = 14, n = 0; m < 32; m++, n++) {
			mac_flag[n] = bit_req_data[m];
		}
		num_addr = mac_flag.to_ulong() / 8;
		bitset<18> mac_block(num_addr);
		mac_flag.reset();
		for (int m = 0, n = 0; n < 14; m++, n++) {
			mac_flag[n] = mac_offset[m];
		}
		for (int m = 0, n = 14; n < 32; m++, n++) {
			mac_flag[n] = mac_block[m];
		}
		addr_mac = Conversion_hex(mac_flag.to_ulong() + MAC_DATA_BEGIN);
		//*****CTR*****//
		bitset<32> ctr_flag(0);
		for (int m = 0, n = 0; m < 14; m++, n++) {
			ctr_flag[n] = bit_req_data[m];
		}
		num_addr = ctr_flag.to_ulong() % 4;
		bitset<14> ctr_offset(num_addr);

		ctr_flag.reset();
		for (int m = 14, n = 0; m < 32; m++, n++) {
			ctr_flag[n] = bit_req_data[m];
		}
		num_addr = ctr_flag.to_ulong() / 4;
		bitset<18> ctr_block(num_addr);
		ctr_flag.reset();
		for (int m = 0, n = 0; n < 14; m++, n++) {
			ctr_flag[n] = ctr_offset[m];
		}
		for (int m = 0, n = 14; n < 32; m++, n++) {
			ctr_flag[n] = ctr_block[m];
		}
		addr_ctr = Conversion_hex(ctr_flag.to_ulong() + CTR_DATA_BEGIN);
		//*****BMT*****//
		string str_temp = addr_ctr;
		const long bmt_data_begn[6] = { 0,917504,974848,978432,978656,978677 };
		for (int index_i = 0; index_i < 6; index_i++)
		{
			strcpy(op_addr, (str_temp.substr(0, 10)).c_str());
			bitset<32> bit_req_data1(strtoul(op_addr, NULL, 16));
			bitset<32> bmt_flag(0);
			for (int m = 0, n = 0; m < 14; m++, n++) {
				bmt_flag[n] = bit_req_data1[m];
			}
			num_addr = bmt_flag.to_ulong() / 16 % (64 / 8);
			bitset<14> bmt_offset(num_addr);

			bmt_flag.reset();
			for (int m = 14, n = 0; m < 32; m++, n++) {
				bmt_flag[n] = bit_req_data1[m];
			}
			num_addr = bmt_flag.to_ulong() / 16 / (64 / 8);
			bitset<18> bmt_block(num_addr);
			bmt_flag.reset();
			for (int m = 0, n = 0; n < 14; m++, n++) {
				bmt_flag[n] = bmt_offset[m];
			}
			for (int m = 0, n = 14; n < 32; m++, n++) {
				bmt_flag[n] = bmt_block[m];
			}
			string str_temp2 = Conversion_hex(bmt_flag.to_ullong() + bmt_data_begn[index_i]);
			addr_bmt.push_back(str_temp2);
			str_temp = str_temp2;
			//cout << str_temp<<"   "<< bmt_data_row [index_i]<< endl;
		}

	}


}  // namespace dramsim3

