// Created the 4th of november of 2019 by Francisco Munoz Martinez

#include "CollectionBusLine.h"
#include "utility.h"
#include <iomanip>


CollectionBusLine::CollectionBusLine(id_t id, std::string name, unsigned int busID, unsigned int input_ports_bus_line, unsigned int connection_width, unsigned int fifo_size) : Unit(id, name) {
	this->input_ports = input_ports_bus_line;
	this->busID = busID;
	//Creating the connections for this bus line
	for (int i = 0; i < this->input_ports; i++) {
		//Adding the input connection
		Connection* input_connection = new Connection(connection_width);
		input_connections.push_back(input_connection);

		//Adding the input fifo
		Fifo* fifo = new Fifo(fifo_size);
		input_fifos.push_back(fifo);

		//Creating the output connection
		output_port = new Connection(connection_width);
		this->collectionbuslineStats.n_inputs_receive.push_back(0); //To track information

	}
	this->store_instruction = 0;
	next_input_selected = 0;
	//  std::cout << "SIZE DESDE EL COLLECTIONBUSLINE: " << this->input_ports << std::endl;
}

CollectionBusLine::~CollectionBusLine() {
	//First removing the input_connections
	for (int i = 0; i < input_connections.size(); i++) {
		delete input_connections[i];
	}

	//Deleting the input_fifos
	for (int i = 0; i < input_fifos.size(); i++) {
		delete input_fifos[i];
	}

	//Deleting output connection
	delete output_port;
}

//added by wangjs
string dec2hex_CB(long long int i) //将int转成16进制字符串
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
//added end

Connection* CollectionBusLine::getInputPort(unsigned int inputID) {
	return this->input_connections[inputID];
}
unsigned long long clk_from_dram_last = 0;
unsigned long long clk_from_dram_now = 0;
void CollectionBusLine::receive() {
	for (int i = 0; i < this->input_connections.size(); i++) {
		if (input_connections[i]->existPendingData()) {
			std::vector<DataPackage*> pck = input_connections[i]->receive();
			for (int j = 0; j < pck.size(); j++) { //Actually this is 1	
				this->collectionbuslineStats.n_inputs_receive[i] += 1; //To track information. Number of packages received by each input line for this output port
				input_fifos[i]->push(pck[j]); //Inserting the package into the fifo
			}
		}
	}
}



void CollectionBusLine::cycle(CPU *cpu) {
 	this->collectionbuslineStats.total_cycles++;


	this->receive(); //Receiving packages from the connections

	bool selected = false;
	unsigned int n_iters = 0;
	//To track Information
	unsigned int n_inputs_trying = 0;
	for (int i = 0; i < input_fifos.size(); i++) {
		if (!input_fifos[i]->isEmpty()) {
			n_inputs_trying += 1;
		}
	}

	this->collectionbuslineStats.n_conflicts_average += n_inputs_trying; //Later this will be divided by the number of total cycles to calculate the average
	if (n_inputs_trying > 1) {
		this->collectionbuslineStats.n_times_conflicts += 1; //To track information
	}

	//End to track information and the actual code to perform the cycle is executed

	std::vector<DataPackage*> data_to_send;
	while (!selected && (n_iters < input_fifos.size())) { //if input not found or there is still data to look up
		if (!input_fifos[next_input_selected]->isEmpty()) { //If there is data in this input then
			selected = true;
			DataPackage* pck = input_fifos[next_input_selected]->pop(); //Poping from the fifo
			pck->setOutputPort(this->busID); //Setting tracking information to the package
			//std::cout << "to bus" << std::endl;

			//added by wangjs
			pck->setAdded_cycle(std::to_string(this->collectionbuslineStats.total_cycles));
			pck->setReq_type("WRITE");
			pck->setAddr_to_dram(dec2hex_CB((long long int)(&input_fifos[next_input_selected])));
			pck->setAddr_type("feature_map");
			pck->setReq_source("npu");
			//added end

			data_to_send.push_back(pck); //Sending the package to memory
			this->collectionbuslineStats.n_sends++; //To track information
		}
		next_input_selected = (next_input_selected + 1) % input_fifos.size();
		n_iters++;
	}

	//Sending the data to the output connection
	if (selected) {

		this->output_port->send(data_to_send);
	}

}

void CollectionBusLine::printStats(std::ofstream& out, unsigned int indent) {
	out << ind(indent) << "{" << std::endl; //TODO put ID
	this->collectionbuslineStats.print(out, indent + IND_SIZE);
	out << ind(indent + IND_SIZE) << ",\"input_fifos_stats\" : [" << std::endl;
	for (int i = 0; i < input_fifos.size(); i++) {
		out << ind(indent + IND_SIZE + IND_SIZE) << "{" << std::endl;
		input_fifos[i]->printStats(out, indent + IND_SIZE + IND_SIZE + IND_SIZE);
		out << ind(indent + IND_SIZE + IND_SIZE) << "}";
		if (i < (input_fifos.size() - 1)) {
			out << ",";
		}

		out << std::endl;
	}
	out << ind(indent + IND_SIZE) << "]" << std::endl;
	out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void CollectionBusLine::printEnergy(std::ofstream& out, unsigned int indent) {
	/*
		This component prints:
			- The input wires connected to this output wire
			- The input FIFOs to connect every input wire
			- The output wire
	*/

	//Printing input wires
	for (int i = 0; i < input_fifos.size(); i++) {
		Connection* conn = input_connections[i];
		conn->printEnergy(out, indent, "CB_WIRE");
	}

	//Printing input fifos
	for (int i = 0; i < input_fifos.size(); i++) {
		Fifo* fifo = input_fifos[i];
		fifo->printEnergy(out, indent);
	}

	//Printing output wire
	output_port->printEnergy(out, indent, "CB_WIRE");
}

int CollectionBusLine::get_Instruction()
{
	return this->store_instruction;
}

void CollectionBusLine::getInputData(string data)
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
	addr_genel = dec2hex_CB(num_addr);
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
	addr_mac = dec2hex_CB(mac_flag.to_ulong() + MAC_DATA_BEGIN);
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
	//ctr_block_num = num_addr;
	bitset<18> ctr_block(num_addr);
	ctr_flag.reset();
	for (int m = 0, n = 0; n < 14; m++, n++) {
		ctr_flag[n] = ctr_offset[m];
	}
	for (int m = 0, n = 14; n < 32; m++, n++) {
		ctr_flag[n] = ctr_block[m];
	}
	addr_ctr = dec2hex_CB(ctr_flag.to_ulong() + CTR_DATA_BEGIN);
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
		string str_temp2 = dec2hex_CB(bmt_flag.to_ullong() + bmt_data_begn[index_i]);
		addr_bmt.push_back(str_temp2);
		str_temp = str_temp2;
		//cout << str_temp<<"   "<< bmt_data_row [index_i]<< endl;
	}


}
