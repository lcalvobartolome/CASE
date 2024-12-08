import subprocess
import threading
import time
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
def extract_cpu_memory_usage(collection, i):
    # wait for the files to be written
    time.sleep(5)
    
    docker_data_tm_cpu = pd.read_csv(f'mondocker_files/tm_cpu_{collection}_{i}', names=['CPU Usage (%)'])
    docker_data_solr_cpu = pd.read_csv(f'mondocker_files/solr_cpu_{collection}_{i}', names=['CPU Usage (%)'])

    docker_data_tm_mem = pd.read_csv(f'mondocker_files/tm_mem_{collection}_{i}', names=['Memory Usage (MB)'])
    try:
        docker_data_tm_mem['Memory Usage (MB)'] = docker_data_tm_mem['Memory Usage (MB)'].apply(lambda x : float(x.split("GiB")[0])*1073.74)
    except Exception as e:
        print(e)
        print("Data is given in MB")
    
    docker_data_solr_mem = pd.read_csv(f'mondocker_files/solr_mem_{collection}_{i}', names=['Memory Usage (MB)'])
    try:
        docker_data_solr_mem['Memory Usage (MB)'] = docker_data_solr_mem['Memory Usage (MB)'].apply(lambda x : float(x.split("GiB")[0])*1073.74)
    except Exception as e:
        print(e)
        print("Data is given in MB")

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
                "field": "date",
                "start": "2000-01-01T00:00:00Z",
                "end": "2024-12-31T23:59:59Z",
                "gap": "+1YEAR"
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

def monitor_idle_mondocker(duration=3600):
    """
    Monitor idle CPU and memory usage for Solr and TM using mondocker.sh.

    Args:
        duration (int): Duration for monitoring in seconds.

    Returns:
        dict: A dictionary containing mean and std for CPU and memory usage for Solr and TM.
    """
    print(f"-- -- Monitoring Solr and TM idle usage for {duration} seconds...")

    # Commands for monitoring Solr and TM CPU and memory usage
    commands = [
        "mondocker.sh 1 mondocker_files/solr_cpu_idle case-solr CPU",
        "mondocker.sh 1 mondocker_files/solr_mem_idle case-solr MEM",
        "mondocker.sh 1 mondocker_files/tm_cpu_idle case-tm CPU",
        "mondocker.sh 1 mondocker_files/tm_mem_idle case-tm MEM"
    ]

    # Start the monitoring processes
    processes = []
    for command in commands:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)

    # Allow monitoring to run for the specified duration
    time.sleep(duration)

    # Terminate all monitoring processes
    for process in processes:
        os.kill(process.pid, signal.SIGTERM)
    print("-- -- Monitoring completed. Collecting results...")

    # Initialize a dictionary to store the results
    results = {
        "solr": {"cpu": {}, "memory": {}},
        "tm": {"cpu": {}, "memory": {}}
    }

    try:
        # Read and process Solr CPU and memory data
        solr_cpu_data = pd.read_csv("mondocker_files/solr_cpu_idle", names=["CPU Usage (%)"])
        solr_mem_data = pd.read_csv("mondocker_files/solr_mem_idle", names=["Memory Usage (MB)"])
        solr_mem_data["Memory Usage (MB)"] = solr_mem_data["Memory Usage (MB)"].apply(
            lambda x: float(x.split("GiB")[0]) * 1073.74 if "GiB" in str(x) else float(x)
        )

        # Read and process TM CPU and memory data
        tm_cpu_data = pd.read_csv("mondocker_files/tm_cpu_idle", names=["CPU Usage (%)"])
        tm_mem_data = pd.read_csv("mondocker_files/tm_mem_idle", names=["Memory Usage (MB)"])
        tm_mem_data["Memory Usage (MB)"] = tm_mem_data["Memory Usage (MB)"].apply(
            lambda x: float(x.split("GiB")[0]) * 1073.74 if "GiB" in str(x) else float(x)
        )

        # Calculate statistics for Solr
        results["solr"]["cpu"]["mean"] = solr_cpu_data["CPU Usage (%)"].mean()
        results["solr"]["cpu"]["std"] = solr_cpu_data["CPU Usage (%)"].std()
        results["solr"]["memory"]["mean"] = solr_mem_data["Memory Usage (MB)"].mean()
        results["solr"]["memory"]["std"] = solr_mem_data["Memory Usage (MB)"].std()

        # Calculate statistics for TM
        results["tm"]["cpu"]["mean"] = tm_cpu_data["CPU Usage (%)"].mean()
        results["tm"]["cpu"]["std"] = tm_cpu_data["CPU Usage (%)"].std()
        results["tm"]["memory"]["mean"] = tm_mem_data["Memory Usage (MB)"].mean()
        results["tm"]["memory"]["std"] = tm_mem_data["Memory Usage (MB)"].std()

        # Save raw data to CSV files
        solr_cpu_data.to_csv("results_files/solr_idle_cpu.csv", index=False)
        solr_mem_data.to_csv("results_files/solr_idle_memory.csv", index=False)
        tm_cpu_data.to_csv("results_files/tm_idle_cpu.csv", index=False)
        tm_mem_data.to_csv("results_files/tm_idle_memory.csv", index=False)

        # Save summary to a text file
        with open("results_files/idle_baseline_summary.txt", "w") as summary_file:
            summary_file.write(
                f"Solr CPU Usage: {results['solr']['cpu']['mean']:.2f}% ± {results['solr']['cpu']['std']:.2f}%\n"
                f"Solr Memory Usage: {results['solr']['memory']['mean']:.2f} MB ± {results['solr']['memory']['std']:.2f} MB\n"
                f"TM CPU Usage: {results['tm']['cpu']['mean']:.2f}% ± {results['tm']['cpu']['std']:.2f}%\n"
                f"TM Memory Usage: {results['tm']['memory']['mean']:.2f} MB ± {results['tm']['memory']['std']:.2f} MB\n"
                # save also total memory and cpu usage
                f"Total CPU Usage: {results['solr']['cpu']['mean'] + results['tm']['cpu']['mean']:.2f}% ± {np.sqrt(results['solr']['cpu']['std']**2 + results['tm']['cpu']['std']**2):.2f}%\n"
                f"Total Memory Usage: {results['solr']['memory']['mean'] + results['tm']['memory']['mean']:.2f} MB ± {np.sqrt(results['solr']['memory']['std']**2 + results['tm']['memory']['std']**2):.2f} MB\n"
            )

        print("-- -- Baseline data saved.")

    except Exception as e:
        print(f"Error processing idle monitoring data: {e}")

    return results

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
            
            try:
                processes = []
                for command in commands:
                    process = run_command_with_pid(command)
                    processes.append(process)

                print(f"-- -- Running SOLR query {i}")

                # Execute the query
                if i == 1:
                    results[f"solr_query_{i}"][collection]["time"]["mean"], \
                    results[f"solr_query_{i}"][collection]["time"]["std"] = execute_query_1(model_name, core_name, session, n_iter=10)
                elif i == 2:
                    results[f"solr_query_{i}"][collection]["time"]["mean"], \
                    results[f"solr_query_{i}"][collection]["time"]["std"] = execute_query_2(model_name, core_name, session, n_iter=10)
                elif i == 3:
                    results[f"solr_query_{i}"][collection]["time"]["mean"], \
                    results[f"solr_query_{i}"][collection]["time"]["std"] = execute_query_3(core_name, word, session, n_iter=10)
                elif i == 4:
                    results[f"solr_query_{i}"][collection]["time"]["mean"], \
                    results[f"solr_query_{i}"][collection]["time"]["std"] = execute_query_4(model_name, core_name, word, session,n_iter=10)

            finally:
                # Ensure all processes are terminated
                for process in processes:
                    os.kill(process.pid, signal.SIGTERM)
                print(f"-- -- Terminated all processes for SOLR query {i}")
            
            # extract cpu / memory usage
            results[f"solr_query_{i}"][collection]["cpu"]["mean"], \
            results[f"solr_query_{i}"][collection]["memory"]["mean"], \
            results[f"solr_query_{i}"][collection]["cpu"]["std"], \
            results[f"solr_query_{i}"][collection]["memory"]["std"] = extract_cpu_memory_usage(collection, i)
            
        #---------------------------
        # Run PANDAS queries
        #---------------------------
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
            results[query_name][collection]["memory"]["std"] = query_function(*query_args, n_iter=10)
            
    print(results)
    
    # Save results to a json file
    with open("results_files/results.json", "w") as file:
        json.dump(results, file)
    
    # Flatten the results into a list of rows for the DataFrame
    rows = []
    for query, datasets in results.items():
        for dataset, metrics in datasets.items():
            rows.append({
                "Query": query,
                "Dataset": dataset,
                "Time (s)": f"{metrics['time']['mean']:.2f} \(\pm\) {metrics['time']['std']:.2f}",
                "CPU (%)": f"{metrics['cpu']['mean']:.2f} \(\pm\) {metrics['cpu']['std']:.2f}",
                "Memory (MB)": f"{metrics['memory']['mean']:.2f} \(\pm\) {metrics['memory']['std']:.2f}",
            })

    df = pd.DataFrame(rows)
    pivot_df = df.pivot(index="Query", columns="Dataset", values=["Time (s)", "CPU (%)", "Memory (MB)"])
    pivot_df.columns = [f"{col[1]} - {col[0]}" for col in pivot_df.columns]
    latex_table = pivot_df.to_latex(index=True, escape=False, column_format="lcccccccc", multicolumn=True, multicolumn_format='c')

    # Write the LaTeX table to a file
    with open("results_files/results_table.tex", "w") as file:
        file.write(latex_table)

    print("LaTeX table saved to results_table.tex!")
    
    print("-- -- Starting the analysis...")

    # Step 1: Monitor Solr and TM idle baseline
    idle_baseline = monitor_idle_mondocker(duration=3600)

    print(f"Solr CPU Usage (Idle): {idle_baseline['solr']['cpu']['mean']:.2f}% ± {idle_baseline['solr']['cpu']['std']:.2f}%")
    print(f"Solr Memory Usage (Idle): {idle_baseline['solr']['memory']['mean']:.2f} MB ± {idle_baseline['solr']['memory']['std']:.2f} MB")
    print(f"TM CPU Usage (Idle): {idle_baseline['tm']['cpu']['mean']:.2f}% ± {idle_baseline['tm']['cpu']['std']:.2f}%")
    print(f"TM Memory Usage (Idle): {idle_baseline['tm']['memory']['mean']:.2f} MB ± {idle_baseline['tm']['memory']['std']:.2f} MB")
    print("Total CPU and memory usage for Solr and TM during idle monitoring:")
    print(f"Total CPU Usage: {idle_baseline['solr']['cpu']['mean'] + idle_baseline['tm']['cpu']['mean']:.2f}%")
    print(f"Total Memory Usage: {idle_baseline['solr']['memory']['mean'] + idle_baseline['tm']['memory']['mean']:.2f} MB")

if __name__ == "__main__":
    main()