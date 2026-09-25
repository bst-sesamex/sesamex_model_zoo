#ifndef __LIGHTSTEREO_MODEL_META_H__
#define __LIGHTSTEREO_MODEL_META_H__

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct ModelMeta {
    // 输入张量信息
    struct InputTensor {
        std::string name;
        std::string tensor_name;
        std::string ctype;      // RGB888P
        std::string dorder;     // WHD
        std::string dtype;      // UINT8
        int width;
        int height;
        int channels;
        int batch_size;
        std::vector<int> size;
        std::vector<int> pitch;
        std::vector<float> dequant_scale;
    };
    
    // 输出张量信息
    struct OutputTensor {
        std::string name;
        std::string tensor_name;
        std::string ctype;      // DEFAULT
        std::string dorder;     // DWH
        std::string dtype;      // INT8
        int depth;
        int width;
        int height;
        int batch_size;
        std::vector<int> size;
        std::vector<int> pitch;
        std::vector<float> dequant_scale;
    };
    
    int net_batch_size;
    int num_input_tensors;
    int num_output_tensors;
    std::vector<InputTensor> input_tensors;
    std::vector<OutputTensor> output_tensors;
    
    bool LoadFromMetaFile(const std::string& meta_file_path) {
        try {
            std::ifstream file(meta_file_path);
            if (!file.is_open()) {
                std::cerr << "Failed to open meta file: " << meta_file_path << std::endl;
                return false;
            }
            
            json meta_json;
            file >> meta_json;
            
            // 读取基本信息
            net_batch_size = meta_json["net_batch_size"];
            num_input_tensors = meta_json["in_tensors"]["num"];
            num_output_tensors = meta_json["out_tensors"]["num"];
            
            // 读取输入张量信息
            input_tensors.clear();
            for (int i = 0; i < num_input_tensors; i++) {
                InputTensor tensor;
                std::string tensor_key = "tensor_" + std::to_string(i);
                auto& tensor_json = meta_json["in_tensors"][tensor_key];
                
                tensor.name = tensor_json["name"];
                tensor.tensor_name = tensor_json["tensor_name"];
                tensor.ctype = tensor_json["ctype"];
                tensor.dorder = tensor_json["dorder"];
                tensor.dtype = tensor_json["dtype"];
                
                tensor.size = tensor_json["Dim"]["size"].get<std::vector<int>>();
                tensor.pitch = tensor_json["Dim"]["pitch"].get<std::vector<int>>();
                tensor.dequant_scale = tensor_json["dequant_scale"].get<std::vector<float>>();
                
                // 解析维度信息 (WHD格式，即width, height, depth)
                tensor.width = tensor.size[0];
                tensor.height = tensor.size[1];
                tensor.channels = tensor.size[2];
                tensor.batch_size = tensor.size[3];
                
                input_tensors.push_back(tensor);
            }
            
            // 读取输出张量信息
            output_tensors.clear();
            for (int i = 0; i < num_output_tensors; i++) {
                OutputTensor tensor;
                std::string tensor_key = "tensor_" + std::to_string(i);
                auto& tensor_json = meta_json["out_tensors"][tensor_key];
                
                tensor.name = tensor_json["name"];
                tensor.tensor_name = tensor_json["tensor_name"];
                tensor.ctype = tensor_json["ctype"];
                tensor.dorder = tensor_json["dorder"];
                tensor.dtype = tensor_json["dtype"];
                
                tensor.size = tensor_json["Dim"]["size"].get<std::vector<int>>();
                tensor.pitch = tensor_json["Dim"]["pitch"].get<std::vector<int>>();
                tensor.dequant_scale = tensor_json["dequant_scale"].get<std::vector<float>>();
                
                // 解析维度信息 (DWH格式，即depth, width, height)
                tensor.depth = tensor.size[0];
                tensor.width = tensor.size[1];
                tensor.height = tensor.size[2];
                tensor.batch_size = tensor.size[3];
                
                output_tensors.push_back(tensor);
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing meta file: " << e.what() << std::endl;
            return false;
        }
    }
    
    void PrintInfo() const {
        std::cout << "=== Model Meta Information ===" << std::endl;
        std::cout << "Batch size: " << net_batch_size << std::endl;
        std::cout << "Input tensors: " << num_input_tensors << std::endl;
        std::cout << "Output tensors: " << num_output_tensors << std::endl;
        
        for (int i = 0; i < input_tensors.size(); i++) {
            const auto& tensor = input_tensors[i];
            std::cout << "Input[" << i << "]: " << tensor.tensor_name 
                      << " (" << tensor.width << "x" << tensor.height << "x" << tensor.channels << ")"
                      << " " << tensor.dtype << " " << tensor.dorder << std::endl;
        }
        
        for (int i = 0; i < output_tensors.size(); i++) {
            const auto& tensor = output_tensors[i];
            std::cout << "Output[" << i << "]: " << tensor.tensor_name 
                      << " (" << tensor.depth << "x" << tensor.width << "x" << tensor.height << ")"
                      << " " << tensor.dtype << " " << tensor.dorder << std::endl;
        }
    }
};

#endif
