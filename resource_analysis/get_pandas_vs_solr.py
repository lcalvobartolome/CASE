import subprocess
import threading
import requests
import json
import numpy as np
import pandas as pd
import pathlib
import os
import signal
import psutil
import pandas as pd
import time
import threading
from scipy import sparse
import subprocess
import time
import signal
import os

##################
# Pandas queries #
##################
def track_psutil_in_background(cpu_memory_data, stop_event, interval=1):
    process = psutil.Process()
    start_time = time.time()

    while not stop_event.is_set():
        cpu_memory_data.append({
            'Time (s)': time.time() - start_time,
            'CPU Usage (%)': process.cpu_percent(interval=1),
            'Memory Usage (MB)': process.memory_info().rss / (1024 * 1024) 
        })
        time.sleep(interval)
        
def execute_pandas_query_1(corpus_year, n_iter:int=10):
    execution_times = []
    cpu_memory_data_list = []

    for i in range(n_iter):
        cpu_memory_data = []
        stop_event = threading.Event()
        
        tracking_thread = threading.Thread(target=track_psutil_in_background, args=(cpu_memory_data, stop_event))
        tracking_thread.start()

        start_time = time.time()
        df_exploded = corpus_year[(corpus_year.year >= 2000) & (corpus_year.year <= 2024)].explode('thetas')
        facet_data = df_exploded.groupby(['year', 'thetas']).size().reset_index(name='count')
        pivot_table = facet_data.pivot(index='thetas', columns='year', values='count').fillna(0)
        time_elapsed = time.time() - start_time

        execution_times.append(time_elapsed)
        print(f"Execution {i+1} time: {time_elapsed} seconds")

        # Stop the CPU/memory tracking
        stop_event.set()
        tracking_thread.join()

        cpu_memory_data_list.extend(cpu_memory_data)

    # Calculate the average time
    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD time: {std_time}")

    cpu_memory_df = pd.DataFrame(cpu_memory_data_list)
    avg_cpu_usage = cpu_memory_df['CPU Usage (%)'].mean()
    avg_memory_usage = cpu_memory_df['Memory Usage (MB)'].mean()
    std_cpu_usage = cpu_memory_df['CPU Usage (%)'].std()
    std_memory_usage = cpu_memory_df['Memory Usage (MB)'].std()

    print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
    print(f"Average memory usage: {avg_memory_usage:.2f} MB")
    print(f"STD CPU usage: {std_cpu_usage:.2f}%")
    print(f"STD memory usage: {std_memory_usage:.2f} MB")
    return mean_time, std_time, avg_cpu_usage, avg_memory_usage, std_cpu_usage, std_memory_usage

def execute_pandas_query_2(corpus_year, thetas_matrix, n_iter:int=10):
    execution_times = []
    cpu_memory_data_list = []

    for j in range(n_iter):
        cpu_memory_data = []
        stop_event = threading.Event()
        
        # Start tracking CPU and memory in a separate thread
        tracking_thread = threading.Thread(target=track_psutil_in_background, args=(cpu_memory_data, stop_event))
        tracking_thread.start()

        start_time = time.time()
        
        facets = {}

        df_exploded = corpus_year[(corpus_year.year >= 2000) & (corpus_year.year <= 2024)].explode('thetas')
        df_with_weights = df_exploded.join(thetas_matrix, how="inner", on="id_top", lsuffix='_left', rsuffix='_right')
        
        for i in range(thetas_matrix.shape[1]):
            topic = f"t{i}"
            if topic in df_with_weights:
                topic_data = df_with_weights[['year', topic]].dropna()
                topic_data_grouped = topic_data.groupby('year')[topic].sum()
                facets[topic] = topic_data_grouped
        time_elapsed = time.time() - start_time

        execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

        # Stop the CPU/memory tracking
        stop_event.set()
        tracking_thread.join()

        cpu_memory_data_list.extend(cpu_memory_data)

    # Calculate the average time
    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD time: {std_time}")

    cpu_memory_df = pd.DataFrame(cpu_memory_data_list)
    avg_cpu_usage = cpu_memory_df['CPU Usage (%)'].mean()
    avg_memory_usage = cpu_memory_df['Memory Usage (MB)'].mean()
    std_cpu_usage = cpu_memory_df['CPU Usage (%)'].std()
    std_memory_usage = cpu_memory_df['Memory Usage (MB)'].std()

    print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
    print(f"Average memory usage: {avg_memory_usage:.2f} MB")
    print(f"STD CPU usage: {std_cpu_usage:.2f}%")
    print(f"STD memory usage: {std_memory_usage:.2f} MB")
    return mean_time, std_time, avg_cpu_usage, avg_memory_usage, std_cpu_usage, std_memory_usage

def execute_pandas_query_3(corpus_year, word, n_iter:int=10):
    execution_times = []
    cpu_memory_data_list = []

    for i in range(n_iter):
        cpu_memory_data = []
        stop_event = threading.Event()
        
        # Start tracking CPU and memory in a separate thread
        tracking_thread = threading.Thread(target=track_psutil_in_background, args=(cpu_memory_data, stop_event))
        tracking_thread.start()
        
        start_time = time.time()

        #filtered_df = corpus_year[corpus_year['lemmas'].apply(lambda x: word in x)]
        count = corpus_year['lemmas'].str.count(word).sum()

        time_elapsed = time.time() - start_time

        execution_times.append(time_elapsed)
        print(f"Execution {i+1} time: {time_elapsed} seconds")

        # Stop the CPU/memory tracking
        stop_event.set()
        tracking_thread.join()

        cpu_memory_data_list.extend(cpu_memory_data)
        
    # Calculate the average time
    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD time: {std_time}")

    cpu_memory_df = pd.DataFrame(cpu_memory_data_list)
    avg_cpu_usage = cpu_memory_df['CPU Usage (%)'].mean()
    avg_memory_usage = cpu_memory_df['Memory Usage (MB)'].mean()
    std_cpu_usage = cpu_memory_df['CPU Usage (%)'].std()
    std_memory_usage = cpu_memory_df['Memory Usage (MB)'].std()

    print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
    print(f"Average memory usage: {avg_memory_usage:.2f} MB")
    print(f"STD CPU usage: {std_cpu_usage:.2f}%")
    print(f"STD memory usage: {std_memory_usage:.2f} MB")
    return mean_time, std_time, avg_cpu_usage, avg_memory_usage, std_cpu_usage, std_memory_usage

def execute_pandas_query_4(corpus_year, word, n_iter:int=10):
    execution_times = []
    cpu_memory_data_list = []

    for j in range(n_iter):
        cpu_memory_data = []
        stop_event = threading.Event()
        
        # Start tracking CPU and memory in a separate thread
        tracking_thread = threading.Thread(target=track_psutil_in_background, args=(cpu_memory_data, stop_event))
        tracking_thread.start()
        
        facets = {}
        
        start_time = time.time()
        
        filtered_df = corpus_year[corpus_year['lemmas'].apply(lambda x: word in x)]
        df_exploded = filtered_df[(filtered_df.year >= 2000) & (filtered_df.year <= 2024)].explode('thetas')
        
        # Generate the facets for topics t0 to t24
        for i in range(25):
            topic = f"t{i}"
            topic_data = df_exploded[df_exploded['thetas'] == i]
            topic_data_grouped = topic_data.groupby('year').size()
            facets[topic] = topic_data_grouped
            time_elapsed = time.time() - start_time
        
            execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

        # Stop the CPU/memory tracking
        stop_event.set()
        tracking_thread.join()

        cpu_memory_data_list.extend(cpu_memory_data)

    # Calculate the average time
    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD time: {std_time}")

    cpu_memory_df = pd.DataFrame(cpu_memory_data_list)
    avg_cpu_usage = cpu_memory_df['CPU Usage (%)'].mean()
    avg_memory_usage = cpu_memory_df['Memory Usage (MB)'].mean()
    std_cpu_usage = cpu_memory_df['CPU Usage (%)'].std()
    std_memory_usage = cpu_memory_df['Memory Usage (MB)'].std()

    print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
    print(f"Average memory usage: {avg_memory_usage:.2f} MB")
    print(f"STD CPU usage: {std_cpu_usage:.2f}%")
    print(f"STD memory usage: {std_memory_usage:.2f} MB")
    return mean_time, std_time, avg_cpu_usage, avg_memory_usage, std_cpu_usage, std_memory_usage

SOLR_BASE_URL = "http://kumo01:8983/solr"

##################
# Solr queries   #
##################
def parse_memory(x):
    if "GiB" in str(x):
        return float(x.split("GiB")[0])*1073.74
    elif "MiB" in str(x):
        return float(x.split("MiB")[0])
    else:
        return float(x)

def extract_cpu_memory_usage(collection, i):
    # wait for the files to be written
    time.sleep(5)
    
    docker_data_tm_cpu = pd.read_csv(f'mondocker_files/tm_cpu_{collection}_{i}', names=['CPU Usage (%)'])
    print(docker_data_tm_cpu)
    docker_data_solr_cpu = pd.read_csv(f'mondocker_files/solr_cpu_{collection}_{i}', names=['CPU Usage (%)'])
    print(docker_data_solr_cpu)

    docker_data_tm_mem = pd.read_csv(f'mondocker_files/tm_mem_{collection}_{i}', names=['Memory Usage (MB)'])
    docker_data_tm_mem['Memory Usage (MB)'] = docker_data_tm_mem['Memory Usage (MB)'].apply(lambda x: parse_memory(x))
    
    docker_data_solr_mem = pd.read_csv(f'mondocker_files/solr_mem_{collection}_{i}', names=['Memory Usage (MB)'])
    docker_data_solr_mem['Memory Usage (MB)'] = docker_data_solr_mem['Memory Usage (MB)'].apply(lambda x: parse_memory(x))

    total_cpu_usage = docker_data_tm_cpu['CPU Usage (%)'] + docker_data_solr_cpu['CPU Usage (%)']
    total_mem_usage = docker_data_tm_mem['Memory Usage (MB)'] + docker_data_solr_mem['Memory Usage (MB)']
    
    combined_data = pd.DataFrame({
        'Total CPU Usage (%)': total_cpu_usage,
        'Total Memory Usage (MB)': total_mem_usage
    })
        
    print(f"Mean CPU usage: {combined_data['Total CPU Usage (%)'].mean()}")
    print(f"Mean memory usage: {combined_data['Total Memory Usage (MB)'].mean()}")
    print(f"STD CPU usage: {combined_data['Total CPU Usage (%)'].std()}")
    print(f"STD memory usage: {combined_data['Total Memory Usage (MB)'].std()}")
    
    # if std times are nan then return 0
    if np.isnan(combined_data["Total CPU Usage (%)"].std()):
        return combined_data["Total CPU Usage (%)"].mean(), combined_data["Total Memory Usage (MB)"].mean(), 0, 0
    else:
        return combined_data["Total CPU Usage (%)"].mean(), combined_data["Total Memory Usage (MB)"].mean(), combined_data["Total CPU Usage (%)"].std(), combined_data["Total Memory Usage (MB)"].std()

def execute_query_1(model_name, core_name, session, n_iter:int=10):
    execution_times = []

    for j in range(n_iter):

        start_time = time.time()

        query = "*:*"
        facet = {
        "topics": {
            "type": "terms",
            "field": model_name,
            "limit": 25,
            "facet": {
            "years": {
                "type": "range",
                #"field": "date",
                #"start": "2000-01-01T00:00:00Z",
                #"end": "2024-12-31T23:59:59Z",
                #"gap": "+1YEAR"
                "type": "terms",
                "field": "year_str",
                "limit": -1,  # To include all years
            }
            }
        }
        }
        query_url = f"{SOLR_BASE_URL}/{core_name}/query"
        params = {
            'q': query,
            'q.op': 'OR',
            'indent': 'true',
            'facet': 'true',
            'json.facet': str(facet).replace("'", '"'),
            'useParams': ''
        }
        
        response = session.get(query_url, params=params)
        
        time_elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            facets = result.get('facets', {})
            #print(f"Facets: {facets}")
        else:
            print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
            facets = {}

        execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD: {std_time}")
    return mean_time, std_time
    
def execute_query_2(model_name, core_name, session, n_iter:int=10):
    execution_times = []

    for j in range(n_iter):

        start_time = time.time()
        
        query = "*:*"
        
        # Generate the facets for topics t0 to t24
        facet = {}
        for i in range(25):
            topic = f"t{i}"
            facet[f"query_facet_{topic}"] = {
                "type": "query",
                "q": f"{model_name}:{topic}",
                "facet": {
                    "years": {
                        "type": "terms",
                        "field": "year_str",
                        "limit": -1,  # To include all years
                        "facet": {
                            "sum_weights": f"sum(payload({model_name},{topic}))".format(topic=topic)
                        }
                    }
                }
            }
        
        """
        facet = {}
        for i in range(25):
            topic = f"t{i}"
            facet[f"query_facet_{topic}"] = {
                "type": "query",
                "q": f"{model_name}:{topic}",
                "facet": {
                    "years": {
                        "type": "range",
                        "field": "date",
                        "start": "2000-01-01T00:00:00Z",
                        "end": "2024-12-31T23:59:59Z",
                        "gap": "+1YEAR",
                        "facet": {
                            "sum_weights": f"sum(payload({model_name},{topic}))".format(topic=topic)    
                        }
                    }
                }
            }
        """
        
        query_url = f"{SOLR_BASE_URL}/{core_name}/query"
        json_payload = {
            'query': query,
            'facet': facet,
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = session.post(query_url, headers=headers, data=json.dumps(json_payload))
        
        time_elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            facets = result.get('facets', {})
            #print(f"Facets: {facets}")
        else:
            print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
            facets = {}

        execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD: {std_time}")
    return mean_time, std_time

def execute_query_3(core_name, word, session, n_iter:int=10):
    execution_times = []

    for j in range(n_iter):

        start_time = time.time()

        query = f"title:{word}"
        
        query_url = f"{SOLR_BASE_URL}/{core_name}/query"
        params = {
            'q': query,
            'q.op': 'OR',
            'indent': 'true',
            'useParams': ''
        }
        
        response = session.get(query_url, params=params)
        
        time_elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            facets = result.get('facets', {})
        else:
            print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
            facets = {}

        execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD: {std_time}")
    return mean_time, std_time

def execute_query_4(model_name, core_name, word, session, n_iter:int=10):
    execution_times = []

    for j in range(n_iter):

        start_time = time.time()

        query = f"title:{word}"
        
        facet = {}
        for i in range(25):
            topic = f"t{i}"
            facet[f"query_facet_{topic}"] = {
                "type": "query",
                "q": f"{model_name}:{topic}", 
                "limit": 30,
            }
        
        query_url = f"{SOLR_BASE_URL}/{core_name}/query"
        json_payload = {
            'query': query,
            'facet': facet,
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = session.post(query_url, headers=headers, data=json.dumps(json_payload))
        
        time_elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            facets = result.get('facets', {})
        else:
            print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
            facets = {}

        execution_times.append(time_elapsed)
        print(f"Execution {j+1} time: {time_elapsed} seconds")

    mean_time = sum(execution_times) / len(execution_times)
    std_time = np.std(execution_times)
    print(f"Average execution time over 10 runs: {mean_time} seconds")
    print(f"STD: {std_time}")
    return mean_time, std_time

def main():
    
    # kill all previous mondocker processes
    os.system("pkill mondocker")
    # remove all previous monitoring files
    os.system("rm -f mondocker_files/solr_cpu_* mondocker_files/solr_mem_* mondocker_files/tm_cpu_* mondocker_files/tm_mem_*")
    # check if the files were removed and stop the execution if not
    if len(os.listdir("mondocker_files")) != 0:
        print("Error removing previous monitoring files. Please check the directory.")
        return
    
    print("-- -- Starting the analysis...")
    
    ## Models
    path_cancer = pathlib.Path("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/WP6models_old/TMmodels/OA_cancer_25tpc")
    path_ai = pathlib.Path("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/WP6models_old/TMmodels/OA_Kwds3_AI_25tpc")

    ## Corpus
    corpusFile_cancer = pathlib.Path("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/WP6models_old/TMmodels/OA_cancer_10tpc/corpus.txt")
    corpusFile_ai = pathlib.Path("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/WP6models_old/TMmodels/OA_Kwds3_AI_25tpc/corpus.txt")
    
    ## Years
    path_years_cancer = pathlib.Path("/export/usuarios_ml4ds/jarenas/cancer_OA_year.parquet")
    path_years_ai = pathlib.Path("/export/usuarios_ml4ds/jarenas/AI_OA_year.parquet")


    def run_command_with_pid(command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    
    # dictionaries to store the results
    queries = ["pandas_query", "solr_query"]
    categories = ["ai", "cancer"]
    num_queries = 4

    # Generate the structure with placeholders for mean and std
    results = {
        f"{query}_{i}": {
            category: {
                "time": {"mean": None, "std": None},
                "cpu": {"mean": None, "std": None},
                "memory": {"mean": None, "std": None}
            } for category in categories
        }
        for query in queries
        for i in range(1, num_queries + 1)
    }
    
    for collection in ["ai", "cancer"]:
        print(f"-- -- Running queries for collection: {collection}")
        if collection == "cancer":
            path2model = path_cancer
            corpusFile = corpusFile_cancer
            yearFile = path_years_cancer

            core_name = "oa_cancer_" 
            model_name = "doctpc_oa_cancer_25tpc"
            word = "metastasis"
        else:
            path2model = path_ai
            corpusFile = corpusFile_ai
            yearFile = path_years_ai

            core_name = "oa_kwds3_ai_" 
            model_name = "doctpc_oa_kwds3_ai_25tpc"
            word = "deep learning"
            
        #---------------------------
        # Run SOLR queries
        #---------------------------
        print(f"-- -- Running SOLR queries for collection: {collection}")
        
        # Reset Solr cache to ensure that the queries are not cached
        # curl "http://localhost:8983/solr/admin/cores?action=RELOAD&core=oa_cancer__shard1_replica_n1"
        # with requests
        cores_to_reload = ["oa_cancer_", "oa_kwds3_ai_", "oa_cancer_25tpc", "oa_kwds3_ai_25tpc"]
        for core_to_reload in cores_to_reload:
            curl_command = f"{SOLR_BASE_URL}/admin/cores?action=RELOAD&core={core_to_reload}_shard1_replica_n1"
            response = requests.get(curl_command)

            print(f"CURL command: {curl_command}")
            print(f"-- -- Resetting Solr cache for core {core_to_reload}_shard1_replica_n1")
            print(f"-- -- Status code: {response.status_code}")
            
        # Wait for Solr to reload the cores. Print while waiting
        wait_time = 10
        for i in range(wait_time):
            print(f"Waiting for Solr to reload cores. {wait_time-i} seconds remaining...")
            time.sleep(1)
        
        # Warm-up request
        # Create a session to reuse connections
        session = requests.Session()
        warm_up_url = f"{SOLR_BASE_URL}/{core_name}/query"
        print(f"-- -- Warming up Solr for collection: {collection}")
        print(f"-- -- Warm-up URL: {warm_up_url}")
        warm_up_params = {'q': '*:*', 'rows': 1}
        response = session.get(warm_up_url, params=warm_up_params)
        if response.status_code == 200:
            print("Warm-up request successful.")
        else:
            print(f"Warm-up failed with status code: {response.status_code}")

        # Run each query while tracking CPU and memory usage
        for i in range(1, num_queries + 1):
            commands = [
                f"mondocker.sh 1 mondocker_files/solr_mem_{collection}_{i} case-solr MEM",
                f"mondocker.sh 1 mondocker_files/solr_cpu_{collection}_{i} case-solr CPU",
                f"mondocker.sh 1 mondocker_files/tm_cpu_{collection}_{i} case-tm CPU",
                f"mondocker.sh 1 mondocker_files/tm_mem_{collection}_{i} case-tm MEM"
            ]

            processes = []
            try:
                for command in commands:
                    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
                    processes.append(process)
                    print(f"Started process: {command}")

                print(f"-- -- Running SOLR query {i}")

                query_fn = globals().get(f"execute_query_{i}")
                if query_fn:
                    start_time = time.time()
                    if i == 4:  # Special handling for execute_query_4
                        mean_time, std_time = query_fn(model_name, core_name, word, session, n_iter=10000)
                    elif i == 3:
                        mean_time, std_time = query_fn(core_name, word, session, n_iter=10000)
                    else:
                        mean_time, std_time = query_fn(model_name, core_name, session, n_iter=1000)
                    elapsed_time = time.time() - start_time
                    print(f"Query {i} completed in {elapsed_time:.2f} seconds.")
                else:
                    print(f"Query function for query {i} not found.")

                time.sleep(5)

            finally:
                for process in processes:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)  
                    process.wait() 
                    print(f"Terminated process with PID {process.pid}")
                print(f"-- -- Terminated all processes for SOLR query {i}")

            mean_cpu, mean_mem, std_cpu, std_mem = extract_cpu_memory_usage(collection, i)
            results[f"solr_query_{i}"][collection] = {
                "cpu": {"mean": mean_cpu, "std": std_cpu},
                "memory": {"mean": mean_mem, "std": std_mem},
                "time": {"mean": mean_time, "std": std_time}
            }
            
            # wait between queries to avoid overloading the system (5 minutes)
            time.sleep(300)


        #---------------------------
        # Run PANDAS queries
        #---------------------------
        """
        print(f"-- -- Running PANDAS queries for collection: {collection}")
        thetas_LDA = sparse.load_npz(path2model / "TMmodel/thetas.npz")
        thetas = thetas_LDA.toarray()
        thetas_matrix = pd.DataFrame(thetas, columns=[f't{i}' for i in range(thetas.shape[1])])
        thetas_matrix["id_top"] = range(len(thetas_matrix))
        
        with open(corpusFile) as fin:
            els = [el for el in fin.readlines()]
        ids = [el.split()[0].strip() for el in els]
        lemmas = [el.split()[2:] for el in els]
        corpus_info = pd.DataFrame(
            {
                "id": ids,
                "lemmas": lemmas
            }
        )
        years_info = pd.read_parquet(yearFile)
        corpus_year = corpus_info.merge(years_info, how="inner", on="id")
        valid_years_mask = (corpus_year["year"] >= 1677) & (corpus_year["year"] <= 2262)
        valid_corpus_year = corpus_year[valid_years_mask]
        corpus_year = valid_corpus_year
        corpus_year["id_top"] = range(len(corpus_year))

        def get_thetas(row, thetas):
            row_data = thetas.getrow(row).toarray().flatten()
            return [id_ for id_, el in enumerate(row_data) if el != 0.0]
        corpus_year["thetas"] = corpus_year.apply(lambda row: get_thetas(row['id_top'], thetas_LDA), axis=1)
        print("-- -- Data loaded")
        
        # Run each query and store the results        
        pandas_queries = {
            "pandas_query_1": (execute_pandas_query_1, [corpus_year]),
            "pandas_query_2": (execute_pandas_query_2, [corpus_year, thetas_matrix]),
            "pandas_query_3": (execute_pandas_query_3, [corpus_year, word]),
            "pandas_query_4": (execute_pandas_query_4, [corpus_year, word])
        }

        for query_name, (query_function, query_args) in pandas_queries.items():
            results[query_name][collection]["time"]["mean"], \
            results[query_name][collection]["time"]["std"], \
            results[query_name][collection]["cpu"]["mean"], \
            results[query_name][collection]["memory"]["mean"], \
            results[query_name][collection]["cpu"]["std"], \
            results[query_name][collection]["memory"]["std"] = query_function(*query_args, n_iter=1000)
            # wait between queries to avoid overloading the system (5 minutes)
            time.sleep(300)
        """
            
    print(results)
    
    # Save results to a json file
    with open("results_files/results.json", "w") as file:
        json.dump(results, file)

if __name__ == "__main__":
    main()