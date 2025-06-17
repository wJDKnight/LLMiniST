from openai import OpenAI
import pandas as pd
import os
import re
import time
from utils import check_batch_status, load_config

# get the path of config file from the command line
import sys
config_path = sys.argv[1]
config = load_config(config_path)

# when read the general config file, the data_name may be specified in the command line
if len(sys.argv) > 2:
    config.data_name = sys.argv[2]
    config.folder_path = f"./batch_json/{config.data_name}_{config.model_type}"
    config.output_path = f"./batch_results/{config.data_name}_{config.model_type}"
    if len(sys.argv) > 3:
        config.replicate = sys.argv[3]

print(config)

if len(sys.argv) > 4:
    parallel_run = sys.argv[4]
else:
    parallel_run = "None"
print(parallel_run)

if not os.path.exists(config.output_path):
    # Create the folder if it doesn't exist
    os.makedirs(config.output_path)


client = OpenAI()
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY_YALE"])

pattern = re.compile(rf"{config.data_name}_(\d+)_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}\.json")
# 遍历文件夹中的所有文件
for filename in os.listdir(config.folder_path):
    # 使用正则表达式提取编号
    match = pattern.match(filename)
    if match:
        # 提取数字并转换为整数
        number = int(match.group(1))

        # # in case of the batch k is not finished
        # if number == 2:
        #     continue
        
        print(f"runing the {filename}")
        batch_input_file = client.files.create(
          file=open(f"{config.folder_path}/{filename}", "rb"),
          purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        endpoint_url = "/v1/chat/completions"  # if model_type == "end2end" else "/v1/embeddings"
        submit_batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=endpoint_url,
            completion_window="24h",
            metadata={
              "description": f"{config.model_type = }. {config.data_name}_batch{number}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}"
            }
        )
        batch_id = submit_batch.id
        print(client.batches.retrieve(submit_batch.id))

        if parallel_run!="parallel":
            s = check_batch_status(client, batch_id)
            if s == "completed":
                file_response = client.files.content(client.batches.retrieve(submit_batch.id).output_file_id)
                save_name = f"response_{config.data_name}_{number}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.txt"
                output_file_name = f"{config.output_path}/{save_name}"
                # Open the file in write mode and save the string
                with open(output_file_name, 'w') as file:
                    file.write(file_response.text)  
                print(f"The response has been saved to {output_file_name}")
            else:
                print(f"Failed with {s}, batch id: {submit_batch.id}")
        else:
            # save batch_id for gathering results later
            # create a txt file named with config.data_name+ config.replicate + time. if the file exists, open it and add batch id as a new line in the end of it. 
            timestamp = time.strftime("%Y%m%d_%H")
            batch_id_file = f"outs/{config.data_name}{config.replicate}_{timestamp}.txt"
            
            # Open file in append mode ('a') - creates file if it doesn't exist
            with open(batch_id_file, 'a') as f:
                f.write(f"{batch_id}\n")
            
            print(f"Batch ID {batch_id} saved to {batch_id_file}")
            