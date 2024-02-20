import psutil


def check_disk_io_counters_support():
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io is None:
            print("disk_io_counters() is not supported on this machine.")
        else:
            print("disk_io_counters() is supported on this machine.")
            print("Disk I/O stats:", disk_io)
    except Exception as e:
        print(f"An error occurred: {e}")

check_disk_io_counters_support()
