#!/usr/bin/env bash

set -uo pipefail

if (( $# == 0 )); then
  echo "usage: run-m5-acceptance.sh COMMAND [ARG ...]" >&2
  exit 2
fi

meminfo_path=${M5_MEMINFO_PATH:-/proc/meminfo}
disk_path=${M5_DISK_PATH:-newcalibre}
monitor_interval_seconds=${M5_MONITOR_INTERVAL_SECONDS:-1}

aggregate_rss_kib() {
  ps -e -o pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1] = $2; rss[$1] = $3 }
    END {
      included[root] = 1
      changed = 1
      while (changed) {
        changed = 0
        for (pid in parent) {
          if (!included[pid] && included[parent[pid]]) {
            included[pid] = 1
            changed = 1
          }
        }
      }
      for (pid in included) total += rss[pid]
      print total + 0
    }
  '
}

memory_headroom_kib() {
  awk '/MemAvailable:/ { print $2 }' "$meminfo_path"
}

free_disk_kib() {
  df -Pk "$disk_path" | awk 'NR == 2 { print $4 }'
}

started_at=$(date +%s)
min_memory_headroom_kib=$(memory_headroom_kib)
min_free_disk_kib=$(free_disk_kib)
peak_process_rss_kib=0

"$@" &
test_pid=$!
while kill -0 "$test_pid" 2>/dev/null; do
  process_rss_kib=$(aggregate_rss_kib "$test_pid")
  memory_headroom=$(memory_headroom_kib)
  free_disk=$(free_disk_kib)
  if (( process_rss_kib > peak_process_rss_kib )); then
    peak_process_rss_kib=$process_rss_kib
  fi
  if (( memory_headroom < min_memory_headroom_kib )); then
    min_memory_headroom_kib=$memory_headroom
  fi
  if (( free_disk < min_free_disk_kib )); then
    min_free_disk_kib=$free_disk
  fi
  sleep "$monitor_interval_seconds"
done

test_status=0
wait "$test_pid" || test_status=$?
elapsed_seconds=$(( $(date +%s) - started_at ))

echo "M5 acceptance elapsed seconds: $elapsed_seconds"
echo "M5 aggregate peak job memory (process RSS) KiB: $peak_process_rss_kib"
echo "M5 minimum memory headroom KiB: $min_memory_headroom_kib"
echo "M5 minimum free disk KiB: $min_free_disk_kib"

sizing_status=0
minimum_headroom_kib=$(( 4 * 1024 * 1024 ))
maximum_elapsed_seconds=$(( 20 * 60 ))
if (( min_memory_headroom_kib < minimum_headroom_kib )); then
  echo "M5 acceptance requires at least 4 GiB memory headroom"
  sizing_status=1
fi
if (( min_free_disk_kib < minimum_headroom_kib )); then
  echo "M5 acceptance requires at least 4 GiB free disk"
  sizing_status=1
fi
if (( elapsed_seconds > maximum_elapsed_seconds )); then
  echo "M5 acceptance must complete within 20 minutes"
  sizing_status=1
fi

echo "M5 acceptance child status: $test_status"
echo "M5 acceptance sizing status: $sizing_status"
if (( test_status != 0 )); then
  exit "$test_status"
fi
exit "$sizing_status"
