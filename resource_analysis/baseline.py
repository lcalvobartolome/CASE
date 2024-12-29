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

if __name__ == "__main__":
    # Monitor idle CPU and memory usage for Solr and TM
    baseline_data = monitor_idle_mondocker(duration=3600)
    print(baseline_data)