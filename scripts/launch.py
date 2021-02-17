#!/usr/bin/env python

from __future__ import print_function

import argparse
import getpass
import os
import psutil
import subprocess
import sys
import time

OUTFILE = 'final.yml'
USER_OPTS = {'ian': '/datasets/home/29/629/ipegg',
             'matheus': '/datasets/home/71/371/mgorski'}


def _spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor


spinner = _spinning_cursor()


def spin():
    sys.stdout.write(next(spinner))
    sys.stdout.flush()
    time.sleep(0.1)
    sys.stdout.write('\b')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, required=True,
                        choices=USER_OPTS,
                        help="Choose a user to pick the home directory")
    parser.add_argument('--name', type=str, default='dev', help="Pod name (default: dev)")
    parser.add_argument('--memory', type=int, default=8, help="Memory in Gb (default 8)")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs (default 1)")
    parser.add_argument('--cpus', type=int, default=2, help="Number of CPUs (default 2)")
    parser.add_argument('--args', type=str, nargs='+', default=["sleep", "infinity"],
                        help="Arguments to be passed to bash (default: sleep infinity)")
    args = parser.parse_args()
    name = args.name
    working_dir = USER_OPTS[args.user]

    with open('pod.yml', 'r') as f:
        config = f.read()
        config = config.format(
            working_dir=working_dir,
            user=args.user,
            name=name,
            memory=args.memory,
            cpus=args.cpus,
            gpus=args.gpus,
            args=args.args,
        )

    with open(OUTFILE, 'w') as f:
        f.write(config)

    subprocess.call(['kubectl', 'create', '-f', OUTFILE])
    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)

    status = b''
    while status.decode('utf-8') != 'Running\n':
        process = subprocess.Popen(
            ['kubectl', 'get', 'pods', name, '-o', 'custom-columns=STATUS:.status.phase', '--no-headers=TRUE'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        status, err = process.communicate()
        spin()
    print("Running")

    # kill running port forwarding processes
    for p in psutil.process_iter():
        if p.username() == getpass.getuser():
            if 'kubectl' in p.name():
                if 'port-forward' in p.cmdline():
                    p.kill()

    ssh_proc = subprocess.Popen(['kubectl', 'port-forward', name, '6666'])  # ssh forwarding
    jupyter_proc = subprocess.Popen(['kubectl', 'port-forward', name, '8888:10629'])  # jupyter forwarding

    subprocess.call(['kubectl', 'exec', name, '-it', '--', 'bash'])
