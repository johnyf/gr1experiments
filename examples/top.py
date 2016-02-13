#!/usr/bin/env python
"""See what python scripts are running."""
import argparse
import datetime
import psutil
import humanize


def main(name):
    me = psutil.Process()
    mypid = me.pid
    rss_all = list()
    vms_all = list()
    zombies = 0
    for proc in psutil.process_iter():
        s = proc.name()
        if s != name:
            continue
        if proc.status() == psutil.STATUS_ZOMBIE:
            zombies += 1
            continue
        pid = proc.pid
        if pid == mypid:
            continue
        aff = proc.cpu_affinity()
        cpu100 = proc.cpu_percent()
        user, system = proc.cpu_times()
        user_str = str(datetime.timedelta(seconds=user))
        system_str = str(datetime.timedelta(seconds=system))
        rss, vms = proc.memory_info()
        rss_str = humanize.naturalsize(rss)
        vms_str = humanize.naturalsize(vms)
        m100 = proc.memory_percent()
        ppid = proc.ppid()
        info = [
            'CPU affinity: {aff}'.format(aff=aff),
            'CPU percent: {x}'.format(x=cpu100),
            'user time: {t}'.format(t=user_str),
            'system time: {t}'.format(t=system_str),
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
    print('found {n} active instances of "{p}"'.format(n=n, p=name))
    print('zombies {z}'.format(z=zombies))
    print('total RSS: {m}'.format(m=total_rss_str))
    print('total VMS: {m}'.format(m=total_vms_str))
    print('total memory: {m}'.format(m=total_memory_str))


def kill(name):
    if name is None:
        print('name is None')
        return
    n = 0
    for proc in psutil.process_iter():
        s = proc.name()
        if s != name:
            continue
        proc.kill()
        n += 1
    print('killed {n} instances of "{name}"'.format(n=n, name=name))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name', default='python', type=str,
                   help='program name', nargs='?')
    p.add_argument('--kill', default=None, type=str,
                   help='terminate all instances')
    args = p.parse_args()
    main(args.name)
    kill(args.kill)

