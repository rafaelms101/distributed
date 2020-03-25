#include "config.h"

#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>

namespace pt = boost::property_tree;

Config cfg;

void Config::loadConfig(std::string filename) {	
	pt::ptree tree;
	pt::read_json(filename, tree);
	cfg.d = tree.get<long>("dimensions");
	cfg.index_path = tree.get<std::string>("index_path");
	cfg.queries_path = tree.get<std::string>("queries_path");
	cfg.distinct_queries = tree.get<long>("distinct_queries");
	cfg.gnd_path = tree.get<std::string>("gnd_path");
	cfg.dataset_size_reduction = tree.get<long>("dataset_size_reduction");
	
	if (cfg.gnd_path.size() == 0) cfg.show_recall = false;
}
