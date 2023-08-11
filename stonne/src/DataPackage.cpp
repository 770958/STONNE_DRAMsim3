//Created 13/06/2019

#include "DataPackage.h"
#include <assert.h>
#include <string.h>

//General constructor implementation

//added by wangjs
DataPackage::DataPackage()
{
	this->added_cycle_to_dram = "added_cycle_to_dram";
	this->addr_to_dram = "addr_to_dram";
	this->req_type = "req_type";
	this->addr_type = "addr_type";
	this->req_source = "req_source";
}
//added end

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source) {
    this->size_package = size_package;
    this->data = data;
    this->data_type =data_type;
    this->source = source;
    this->traffic_type = UNICAST; //Default
	this->delay = 0;
	this->added_cycle_to_dram = "added_cycle_to_dram";
	this->addr_to_dram = "addr_to_dram";
	this->req_type = "req_type";
	this->addr_type = "addr_type";
	this->req_source = "req_source";
}

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type) : DataPackage(size_package, data, data_type, source) {
    this->traffic_type = traffic_type;
    this->dests = NULL;
}
// Unicast package constructor. 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest) : 
DataPackage(size_package, data, data_type, source, traffic_type) {
    assert(traffic_type == UNICAST);
    this->unicast_dest = unicast_dest;
}
//Multicast package. dests must be dynamic memory since the array is not copied. 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, bool* dests, unsigned int n_dests) : DataPackage(size_package, data, data_type, source, traffic_type) {
    this->dests = dests;
    this->n_dests = n_dests;
}

//psum package
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int VN, adderoperation_t operation_mode): DataPackage(size_package, data, data_type, source) {
    this->VN = VN;
    this->operation_mode = operation_mode;
}

unsigned long long DataPackage::getDelay()
{
   
	return this->delay; 
}

void DataPackage::setOutputPort(unsigned int output_port) {
    this->output_port = output_port;
}

void DataPackage::setIterationK(unsigned int iteration_k) {
    this->iteration_k = iteration_k;
}

//set delay
void DataPackage::setDelay(unsigned long long delay)
{
	this->delay = delay ;
}



//Copy constructor
DataPackage::DataPackage(DataPackage* pck) {
    this->size_package = pck->get_size_package();
    this->data = pck->get_data();
    this->data_type = pck->get_data_type();
    this->source = pck->get_source();
    this->traffic_type = pck->get_traffic_type();
    this->unicast_dest = pck->get_unicast_dest();
    this->VN = pck->get_vn();
    this->operation_mode = pck->get_operation_mode();
    this->output_port = output_port;
    this->iteration_k=pck->getIterationK();
	this->delay = pck->getDelay();
	this->added_cycle_to_dram = pck->getAdded_cycle();
	this->addr_to_dram = pck->getAddr_to_dram();
	this->req_type = pck->getReq_type();
	this->addr_type = pck->getAddr_type();
	this->req_source = pck->getReq_source();
    if(this->traffic_type == MULTICAST) {
        this->n_dests = pck->get_n_dests();  
        const bool* prev_pck_dests = pck->get_dests();
        this->dests = new bool[this->n_dests]; 
        memcpy(this->dests, prev_pck_dests, sizeof(bool)*this->n_dests);

    }
}

DataPackage::~DataPackage() {
  
	//delete by wangjs due to some errors while running the whole system
  /*  if(this->traffic_type==MULTICAST) {
        delete[] dests;
    }*/
}


