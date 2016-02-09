#!/usr/bin/env python
"""See what python scripts are running."""
import psutil
import humanize


def main():
    name = 'python'
    rss_all = list()
    vms_all = list()
    for proc in psutil.process_iter():
        s = proc.name()
        if s != name:
            continue
        print(s)
        aff = proc.cpu_affinity()
        cpu100 = proc.cpu_percent()
        user, system = proc.cpu_times()
        rss, vms = proc.memory_info()
        rss_str = humanize.naturalsize(rss)
        vms_str = humanize.naturalsize(vms)
        m100 = proc.memory_percent()
        pid = proc.pid
        ppid = proc.ppid()
        info = [
            'CPU affinity: {aff}'.format(aff=aff),
            'CPU percent: {x}'.format(x=cpu100),
            'user time: {t}'.format(t=user),
            'system time: {t}'.format(t=system),
            'RSS: {m}'.format(m=rss_str),
            'VMS: {m}'.format(m=vms_str),
            'memory percent: {m}'.format(m=m100),
            'PID: {p}'.format(p=pid),
            'parent PID: {p}'.format(p=ppid)]
        print('\n' + 60 * '-')
        print('\n'.join(info))
        # aggregate
        rss_all.append(rss)
        vms_all.append(vms)
    # sum RSS and VMS
    total_rss = sum(rss_all)
    total_rss_str = humanize.naturalsize(total_rss)
    total_vms = sum(vms_all)
    total_vms_str = humanize.naturalsize(total_vms)
    n = len(rss_all)
    total_memory = psutil.virtual_memory().total
    total_memory_str = humanize.naturalsize(total_memory)
    print('\n' + 60 * '-')
    print('found {n} instances of "{p}"'.format(n=n, p=name))
    print('total RSS: {m}'.format(m=total_rss_str))
    print('total VMS: {m}'.format(m=total_vms_str))
    print('total memory: {m}'.format(m=total_memory_str))


if __name__ == '__main__':
    main()
