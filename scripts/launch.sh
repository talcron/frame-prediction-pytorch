#!/bin/bash

# FD 3= informational messages initially routed to STDERR but suppressible via "-q" quiet flag
exec 3>&2

. /software/common64/dsmlp/bin/kubevars.sh

if [ \! -f $KUBECONFIG ] && sleep 3 && [ \! -f $KUBECONFIG ]; then
	echo "Error; could not access configuration file: $KUBECONFIG." >&2
	echo "Please wait a few seconds and try again, or contact ITS/ETS for assistance." >&2
	exit 1
fi

if [ \! -f /usr/bin/kubectl ]; then
	echo "Error: could not find kubectl binary on host $(hostname)" >&2
	echo "Please try another ieng6-* node, and report the missing package to ITS/ETS." >&2
	exit 1
fi

K8S_NAMESPACE=${K8S_NAMESPACE:-${K8S_USER}}
K8S_HOME_DIR=${K8S_HOME_DIR_MOUNT}

if kubectl get pvc dsmlp-datasets-nfs-temp 2>/dev/null > /dev/null; then
        K8S_HOME_PVC=${K8S_HOME_PVC:-"home-nfs-temp"}
        K8S_DATASETS_PVC=${K8S_DATASETS_PVC:-"dsmlp-datasets-nfs-temp"}
        K8S_SUPPORT_PVC=${K8S_SUPPORT_PVC:-"support-nfs-temp"}
        K8S_TOLERATION=${K8S_TOLERATION:-"supplemental-nodes"}
else
	K8S_HOME_PVC=${K8S_HOME_PVC:-"home"}
	K8S_DATASETS_PVC=${K8S_DATASETS_PVC:-"dsmlp-datasets"}
	K8S_SUPPORT_PVC=${K8S_SUPPORT_PVC:-"support"}
fi

K8S_HOME_DIR=$(echo $K8S_HOME_DIR | sed -e 's#^/home/linux/dsmlp/#/datasets/home/#')

# Generate random Jupyter token for this session
# won't be needed once we implement proper JupyterHub
TMP_TOK=($(dd if=/dev/urandom bs=512 count=1 status=none | sha256sum))
JUPYTER_TOKEN=${TMP_TOK[0]}

# Container to run (and entrypoint/start command for Jupyter server)

if [[ "$PY3" =~ ^[yY][eE][sS]$ ]]; then
	DEF_DOCKER_IMAGE="ucsdets/instructional:ets-pytorch-py3-latest"
else
	DEF_DOCKER_IMAGE="ucsdets/instructional:ets-pytorch-py2-latest"
fi
K8S_DOCKER_IMAGE=${K8S_DOCKER_IMAGE:-${DEF_DOCKER_IMAGE}}
K8S_ENTRYPOINT=${K8S_ENTRYPOINT:-"bash"}

# can't override from environment directly; set K8S_ENTRYPOINT_ARGS_EXPANDED from parent script instead
# (it will be picked up after arg parsing below)
# K8S_ENTRYPOINT_ARGS=("--ip=0.0.0.0" "--NotebookApp.notebook_dir=${K8S_HOME_DIR}" "--NotebookApp.token=${JUPYTER_TOKEN}" "--KernelRestarter.restart_limit=0" "--debug")

K8S_NUM_GPU=${K8S_NUM_GPU:-0}
K8S_NUM_CPU=${K8S_NUM_CPU:-1}
K8S_GB_MEM=${K8S_GB_MEM:-1}

# Default to meaningless "Linux OS" selector 
K8S_NODE_SELECTOR_KEY=${K8S_NODE_SELECTOR_KEY:-"beta.kubernetes.io/os"}
K8S_NODE_SELECTOR_VALUE=${K8S_NODE_SELECTOR_VALUE:-"linux"}

K8S_PRIORITY_CLASS_NAME=${K8S_PRIORITY_CLASS_NAME:-normal}

K8S_GPU_RESOURCE_TYPE=${K8S_GPU_RESOURCE_TYPE:-"nvidia.com/gpu"}

if [ "$K8S_EXTRA_SECURITY_CONTEXT" ]; then
  K8S_DECODED_EXTRA_SECURITY_CONTEXT=$(echo "$K8S_EXTRA_SECURITY_CONTEXT" | base64 --decode)
fi

######
# Additional configuration - can be left as-is

# Environment defaults
K8S_INIT=${K8S_INIT:-"/opt/k8s-support/bin/tini"}
K8S_INITENV=${K8S_INITENV:-"/opt/k8s-support/bin/initenv-createhomedir.sh"}

# Spawn an interactive shell via kubectl exec, and terminate pod on shell exit
SPAWN_INTERACTIVE_SHELL=${SPAWN_INTERACTIVE_SHELL:-YES}

# Set to 0 to inhibit socat/kubectl port-forward
PROXY_ENABLED=${PROXY_ENABLED:-YES}
PROXY_PORT=${PROXY_PORT:-8888}
PROXY_HOSTNAME=`hostname`
PROXY_IP=$(host -t a `hostname` | sed -e 's#^.*has address ##')

# Dump podspec to stdout
DUMP_PODSPEC_STDOUT=no

K8S_TOLERATION=${K8S_TOLERATION:-"NONE"}

########################
# By default no nbgrader mounts into launched pods
PROTO_NBGRADER_MOUNTS=$(cat <<'EOM'
    # no nbgrader mounts
EOM
)
PROTO_NBGRADER_VOLS=$(cat <<'EOM'
    # no nbgrader vols
EOM
)
NBGRADER_MOUNTS=${NBGRADER_MOUNTS:-${PROTO_NBGRADER_MOUNTS}}
NBGRADER_VOLS=${NBGRADER_VOLS:-${PROTO_NBGRADER_VOLS}}
NBGRADER_COURSEID=${NBGRADER_COURSEID:-"-none"}

########################
# Default to old behavior
CUDA_LIBRARY_PATH=${CUDA_LIBRARY_PATH:-"/usr/local/cuda-8.0/targets/x86_64-linux/lib:/usr/local/cuda-8.0/lib64:/usr/lib64/nvidia:/usr/local/cuda/extras/CUPTI/lib64"}
CUDA_LD_LIBRARY_PATH_VARNAME=${CUDA_LD_LIBRARY_PATH_VARNAME:-"LD_LIBRARY_PATH"}
CUDA_LIB_MOUNTS=$(cat <<'EOM'
    - mountPath: /usr/lib64/nvidia
      name: nvidiadrv
    - mountPath: /usr/local/cuda-8.0/lib64
      name: cuda
EOM
)
CUDA_LIB_VOLS=$(cat <<'EOM'
  - name: cuda  # /usr/local/cuda-8.0/lib64
    persistentVolumeClaim:
      claimName:  cuda
  - name: nvidiadrv  #  /usr/lib64/nvidia
    persistentVolumeClaim:
      claimName:  nvidiadrv
EOM
)

if [ "$K8S_CUDA_VERSION" = "9" ]; then
CUDA_LIB_MOUNTS=$(cat <<'EOM'
    - mountPath: /usr/lib64/nvidia
      name: nvidiadrv
EOM
	)
CUDA_LIB_VOLS=$(cat <<'EOM'
  - name: nvidiadrv  #  /usr/lib64/nvidia
    persistentVolumeClaim:
      claimName:  nvidiadrv
EOM
)
   CUDA_LIBRARY_PATH="/usr/lib64/nvidia:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64"
   CUDA_LD_LIBRARY_PATH_VARNAME="LD_LIBRARY_PATH"
fi

#################
# New (>=1.9) GPU allocation config
K8S_GPU_TOLERATION=${K8S_GPU_TOLERATION:-"NONE-GPU"}
if [ "$K8S_GPU_RESOURCE_TYPE" = "nvidia.com/gpu" ]; then
   K8S_GPU_TOLERATION="nvidia.com/gpu"
   CUDA_LIBRARY_PATH="/usr/local/nvidia/lib64"
   CUDA_LD_LIBRARY_PATH_VARNAME="EXTRA_LD_LIBRARY_PATH"
   CUDA_LIB_VOLS="#"
   CUDA_LIB_MOUNTS="#"
fi

#### End configuration section

# Point users to our front-end script if it exists
if [ "$K8S_CONFIG_SOURCE" ]; then
	EXPANDED_CONFIG_SOURCE="$( cd "$( dirname "$K8S_CONFIG_SOURCE" )" && pwd )"/$(basename $K8S_CONFIG_SOURCE)
fi

#### Command-line options to override any ENV-based configuration above
while getopts ":n:hi:c:g:m:dsBbBv:e:p:xqrft" opt; do
  case $opt in
    d) 
	DUMP_PODSPEC_STDOUT=yes
	;;
    B)
      SPAWN_INTERACTIVE_SHELL=NO
      PROXY_ENABLED=NO
      ;;
    s) 
      K8S_ENTRYPOINT="/opt/k8s-support/bin/pause"
      K8S_ENTRYPOINT_ARGS=()
      SPAWN_INTERACTIVE_SHELL=YES
      PROXY_ENABLED=NO
      ;;
    b)
      K8S_INIT="/opt/k8s-support/bin/pause"
      K8S_ENTRYPOINT_ARGS=()
      SPAWN_INTERACTIVE_SHELL=NO
      PROXY_ENABLED=NO
      ;;
    q)
      exec 3>/dev/null
      ;;
    r)
      SPAWN_INTERACTIVE_SHELL=NO
      PROXY_ENABLED=NO
      CAPTURE_BATCH_CMD=YES
      BATCH_JOB=YES
      ;;
    f)
      FOLLOW_LOGS=YES
      ;;
    h)
      echo "Usage: ${K8S_CONFIG_SOURCE} [-n nodenum] [-i dockerepo/image] [-c #core] [-g #gpu] [-m #gb-ram] [-v (gtx1080ti|k5200|titan)] [-b]" >&2
      exit 0
      ;;
    c)
      if ! [[ $OPTARG =~ ^[0-9]+$ ]]; then
         echo "Error: Non-numeric #cpu: $OPTARG" >&2; exit 1
      fi
      K8S_NUM_CPU=$OPTARG
      ;;
    g)
      if ! [[ $OPTARG =~ ^[0-9]+$ ]]; then
         echo "Error: Non-numeric #gpu: $OPTARG" >&2; exit 1
      fi
      K8S_NUM_GPU=$OPTARG
      ;;
    m)
      if ! [[ $OPTARG =~ ^[0-9]+$ ]]; then
         echo "Error: Non-numeric #gb-ram: $OPTARG" >&2; exit 1
      fi
      K8S_GB_MEM=$OPTARG
      ;;
    p)
      if ! [[ $OPTARG =~ ^(low|normal)$ ]]; then
         echo "Error: unknown priority $OPTARG; valid choices: low,normal" >&2; exit 1
      fi
      K8S_PRIORITY_CLASS_NAME=$OPTARG
      ;;
    e)
      K8S_ENTRYPOINT=$OPTARG
      ;;
    i)
      K8S_DOCKER_IMAGE=$OPTARG
      ;;
    v)
      if ! [[ $OPTARG =~ ^(gtx1080ti|k5200|titan)$ ]]; then
         echo "Error: unknown gpu-model $OPTARG; valid choices: gtx1080ti,k5200,titan" >&2; exit 1
      fi
      K8S_NODE_SELECTOR_KEY="gputype"
      K8S_NODE_SELECTOR_VALUE="$OPTARG"
      ;;
    n)
      K8S_NODE_SELECTOR_KEY="kubernetes.io/hostname"
      if [[ $OPTARG =~ ^[0-9]+$ ]]; then
      	K8S_NODE_SELECTOR_VALUE=$(printf "its-dsmlp-n%02d.ucsd.edu" $OPTARG)
      else
      	K8S_NODE_SELECTOR_VALUE=$OPTARG
      fi
      ;;
    t)
	K8S_HOME_PVC="home-nfs-temp"
	K8S_DATASETS_PVC="dsmlp-datasets-nfs-temp"
	K8S_SUPPORT_PVC="support-nfs-temp"
	K8S_TOLERATION="supplemental-nodes"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

########################################
# Process command line args
if [ "$CAPTURE_BATCH_CMD" ]; then
	shift $((OPTIND-1))
	BATCH_CMD=()
	for i in "$@"; do
		BATCH_CMD+=( "$i" )
	done
fi

# A batch job replaces entrypoint and args with commandline args
if [ "$BATCH_JOB" = "YES" ]; then
	K8S_ENTRYPOINT="${BATCH_CMD[0]}"
	K8S_ENTRYPOINT_ARGS=("${BATCH_CMD[@]:1}")
fi

# Leave _ARGS_EXPANDED unset if arg array is empty
# The IFS= business is intended to surround arguments with single-quotes, then join them with a comma
if [ \! "$K8S_ENTRYPOINT_ARGS_EXPANDED" -a ${#K8S_ENTRYPOINT_ARGS[@]} -gt 0 ]; then
	NIFS=$IFS
	IFS=$'\001'
	K8S_ENTRYPOINT_ARGS_EXPANDED=$(echo "${K8S_ENTRYPOINT_ARGS[*]}" | sed -e "s/"$'\001'"/','/g" -e "s/^/'/" -e "s/\$/'/")
	IFS="$NIFS"
fi

##############
# Apply limits
# At some point these will be enforced by Admission controller, but
# imposing parallel checks here can result in more relevant error messages => nicer UX

if [ "$K8S_PRIORITY_CLASS_NAME" = "low" ]; then
	# 48-hour deadline unless otherwise specified
	K8S_TIMEOUT_SECONDS=${K8S_TIMEOUT_SECONDS:-$(( 3600 * 48 ))}
else
	# Absolute maximum lifetime for this job - should be relatively large
	# Default to 6 hours
	K8S_TIMEOUT_SECONDS=${K8S_TIMEOUT_SECONDS:-$(( 3600 * 6 ))}

	if (( $K8S_TIMEOUT_SECONDS > (3600 * 12) )) && [ -z "$K8S_BYPASS_TIMEOUT_LIMIT" ] ; then
		echo "K8S_TIMEOUT_SECONDS may not exceed 12 hours.  Please contact ITS/ETS for assistance." >&2
		exit 1
	fi

	if (( $K8S_NUM_GPU > 2 )) && [ -z "$K8S_BYPASS_GPU_LIMIT" ] ; then
		echo "${k8S_NUM_GPU} exceeds standard GPU limit of 2.  Please contact ITS/ETS for assistance." >&2
		exit 1
	fi
fi

############################################################
# Set default minimum CPU available on a node to 50% requested cores
# (doing so will better utilize idle cycles during interactive sessions.)
K8S_NUM_CPU_REQ=${K8S_NUM_CPU_REQ:-$(bc <<<"scale=2;$K8S_NUM_CPU/2")}

#######################################################
# Gently steer CPU-only jobs towards our CPU-only nodes
if (( $K8S_NUM_GPU == 0 )); then
	GPUTYPE_NONE_AFFINITY_OP="In"
else
	GPUTYPE_NONE_AFFINITY_OP="NotIn"
fi
###

main () {
	# NOTE: This pod name could be more descriptive (e.g. based off container being run)
	POD_NAME="${K8S_USERNAME}-$$"

	if [ "$DUMP_PODSPEC_STDOUT" == "yes" ]; then
		create_pod_spec | cat
		exit 0
	fi

        echo "Attempting to create job ('pod') with ${K8S_NUM_CPU} CPU cores, ${K8S_GB_MEM} GB RAM, and ${K8S_NUM_GPU} GPU units." >&3
	[ "${EXPANDED_CONFIG_SOURCE}" ] && echo "   (Adjust command line options, or edit \"${EXPANDED_CONFIG_SOURCE}\" to change this configuration.)" >&3

	# create_pod takes most of the above global variables as input
	if ! create_pod; then
		echo $(date) "Pod creation failed - contact ITS/ETS for support" >&2
		echo "NOTE: An error \"forbidden: exceeded quota\" may indicate that you have disconnected/background jobs running." >&2
		echo "      Please run \"kubectl get pods\", then run \"kubectl delete pod <pod-id>\" to terminate them." >&2
		exit 1
	fi

	if [ "$BATCH_JOB" = "YES" -a "$FOLLOW_LOGS" != "YES" ]; then
		echo "Batch job submitted; \"kubectl get pods\" to list pending/running pods." >&3
		echo "You may retrieve output from your pod via: \"kubectl logs ${POD_NAME}\"." >&3
		echo "PODNAME=${POD_NAME}"
		exit
        fi

	trap "REASON=trap-EXIT; cleanup" EXIT
	trap "REASON=trap-INT; cleanup" INT

	# FIXME - should timeout after N minutes
	while true; do
		S=$(pod_status)
		if [ "$S" = "Running" ]; then
			IP=$(pod_ip)
			NODE=$(pod_nodename)
			echo $(date) "pod is running with IP: $IP on node: $NODE" >&3
			echo "${K8S_DOCKER_IMAGE} is now active." >&3
			echo "" >&3

			break
		elif [ "$S" = "Failed" -o "$S" = "" ]; then
			kubectl describe pod ${POD_NAME} 1>&2
			kubectl logs ${POD_NAME} 1>&2
			echo $(date) "pod startup failed - contact ITS/ETS for support" 1>&2
			exit 1
		elif [ "$S" = "Succeeded" ]; then
			echo "Success - ${K8S_DOCKER_IMAGE} completed immediately." >&3
			break
		else
			C=`pod_conditions`
			echo $(date) "starting up - pod status: $S ; $C" >&3
			sleep 5
			continue
        	fi
	done

	if [ "$PROXY_ENABLED" = "YES" ]; then 
		SOCAT_PORT=$(findport)
		if [[ ! $SOCAT_PORT ]]; then
			echo $(date) "Could not allocate port forwarding port - contact ITS/ETS for support"
			
			exit 1
		fi
	
		kubectl port-forward ${POD_NAME} ${SOCAT_PORT}:${PROXY_PORT} < /dev/null > /dev/null 2>&1 &
		KUBE_FWD_PID=$!
	        echo pod name: ${POD_NAME}
		echo socat port: ${SOCAT_PORT}
		echo proxy port: ${PROXY_PORT}
		echo proxy ip: ${PROXY_IP}

		socat tcp4-listen:${SOCAT_PORT},reuseaddr,fork,bind=${PROXY_IP} tcp4-connect:127.0.0.1:${SOCAT_PORT} < /dev/null > /dev/null 2>&1 &
		SOCAT_PID=$!

		APP_URL="http://${PROXY_HOSTNAME}:${SOCAT_PORT}/?token=${JUPYTER_TOKEN}"
	
		echo "Please connect to: ${APP_URL}"
		echo ""

		CLEANUP_NOTE=' and close Jupyter notebooks'

	fi

	# If interactive shell is requested, keep pod alive until shell exits
	if [ "$SPAWN_INTERACTIVE_SHELL" = "YES" ]; then
		echo "Connected to ${POD_NAME}; type 'exit' to terminate pod/processes${CLEANUP_NOTE}." >&3
		kubesh ${POD_NAME} 
		REASON="kubesh exit: $?"
		cleanup
	elif [ "$PROXY_ENABLED" = "YES" ]; then
		# keep pod alive until something dies, or user triggers cleanup() via ctrl-c
		while sleep 5; do
			podstat=`pod_status`
			if ! check_socat && check_kube_fwd ; then
				echo $(date) "network proxy has died; terminating pod and cleaning up."
				break
			elif [ "$podstat" = "Succeeded" ]; then
				echo $(date) "container has exited; cleaning up." >&3
				break
			elif [ "$podstat" != "Running" ]; then
				echo $(date) "container terminated; cleaning up." >&3
				break
			fi
		done
	elif [ "$FOLLOW_LOGS" = "YES" ]; then
		kubectl logs -f ${POD_NAME}

		REASON="kubectl logs exit: $?"
		cleanup
	else
		echo "Connect to your background pod via: \"kubesh ${POD_NAME}\"" >&3
		echo "Please remember to shut down via: \"kubectl delete pod ${POD_NAME}\" ; \"kubectl get pods\" to list running pods." >&3
		echo "You may retrieve output from your pod via: \"kubectl logs ${POD_NAME}\"." >&3
		echo "PODNAME=${POD_NAME}"

		# Disable cleanup-on-exit
		trap EXIT 2>/dev/null || true
		trap INT 2>/dev/null || true
	fi

}

function cleanup () {
	syslog "cleanup of ${POD_NAME} after ${REASON}"
	check_socat && kill_socat && sleep 1
	check_kube_fwd && kill_kube_fwd && sleep 1

	reason=$( (pod_container_term_reason ; pod_term_reason) | grep . )
	if [ "$reason" != "" -a "$reason" != "Completed" ]; then
		echo "" 1>&2; echo pod '"'${POD_NAME}'"' "exited due to:" $reason 1>&2
	fi
	
	[ -z "$PRESERVE_POD" ] && kill_pod

	trap EXIT 2>/dev/null || true
	trap INT 2>/dev/null || true

	exit
}

function check_kube_fwd() {
	[ \! -z ${KUBE_FWD_PID} ] && kill -0 ${KUBE_FWD_PID} 2>/dev/null
}

function kill_kube_fwd() {
	kill -QUIT ${KUBE_FWD_PID} 2>/dev/null
}

function check_socat() {
	[ \! -z ${SOCAT_PID} ] && kill -0 ${SOCAT_PID} 2>/dev/null
}

function kill_socat() {
	kill -QUIT ${SOCAT_PID} 2>/dev/null
}

function pod_conditions() {
	kubectl get pod ${POD_NAME} -o jsonpath="{ .status.conditions[0].message }"
}

function pod_status() {
	kubectl get pod ${POD_NAME} -o jsonpath="{ .status.phase }" 2>/dev/null 
}

function pod_term_reason {
	kubectl get pod ${POD_NAME} -o jsonpath="{ .status.reason}"
}

function pod_container_term_reason {
	kubectl get pod ${POD_NAME} -o jsonpath="{ .status.containerStatuses[0].state.terminated.reason}"
}

function pod_nodename() {
	kubectl get pod ${POD_NAME} --template='{{.spec.nodeName}}'
}

function pod_ip() {
	kubectl get pod ${POD_NAME} --template='{{.status.podIP}}'
}

# FIXME - should timeout if presented with unkillable pod!
function kill_pod() {
	while kubectl delete pod ${POD_NAME} 1>&3; do 
		sleep 5
		S=`pod_status`
		if [ "$S" != 'Running' ]; then
			break
		fi
		echo $(date) "Waiting for your processes to terminate..." >&3
	done
}

function create_pod_spec() {
	cat <<EOM
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${K8S_NAMESPACE}
  labels:
    name: ${POD_NAME}
    svcreg: "true"
    preplabel: "${PREPLABEL}"
spec:
  priorityClassName: "${K8S_PRIORITY_CLASS_NAME}"
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 30
        preference:
          matchExpressions:
          - key: gputype
            operator: ${GPUTYPE_NONE_AFFINITY_OP}
            values:
            - none
  nodeSelector:
      ${K8S_NODE_SELECTOR_KEY}: ${K8S_NODE_SELECTOR_VALUE}
  tolerations:
  - key: ${K8S_TOLERATION}
    operator: "Exists"
    effect: "NoSchedule"
  - key: ${K8S_GPU_TOLERATION}
    operator: "Exists"
    effect: "NoSchedule"
  securityContext:
    runAsUser: ${K8S_UID}${K8S_DECODED_EXTRA_SECURITY_CONTEXT}
  hostname: "${POD_NAME}"
  subdomain: "pods"
  initContainers:
  - args:
    - cp -r /support/* /tmp/k8s-support ; ls -al /tmp/k8s-support
    command:
    - /bin/sh
    - -c
    image: ucsdets/k8s-support:6fea9c2
    imagePullPolicy: IfNotPresent
    name: init-support
    resources:
      limits:
        cpu: 500m
        memory: 256M
      requests:
        cpu: 500m
        memory: 256M
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
    - mountPath: /tmp/k8s-support
      name: support
  containers:
  - env:
    - name: NB_USER
      value: ${K8S_USERNAME}
    - name: NB_UID
      value: "${K8S_UID}"
    - name: XDG_CACHE_HOME
      value: "/tmp/xdg-cache"
    - name: USER
      value: ${K8S_USERNAME}
    - name: LOGNAME
      value: ${K8S_USERNAME}
    - name: TERM
      value: xterm
    - name: TZ
      value: "PST8PDT"
    - name: SHELL
      value: /bin/bash
    - name: PATH
      value: /usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    - name: NBGRADER_COURSEID
      value: ${NBGRADER_COURSEID}
    - name: HOME
      value: ${K8S_HOME_DIR}
    - name: PREPLABEL
      value: "${PREPLABEL}"
    - name: JUPYTER_TOKEN
      value: ${JUPYTER_TOKEN}
    - name: ${CUDA_LD_LIBRARY_PATH_VARNAME}
      value: "${CUDA_LIBRARY_PATH}"
    - name: https_proxy
      value: http://web.ucsd.edu:3128
    - name: http_proxy
      value: http://web.ucsd.edu:3128
    - name: KUBERNETES_NODE_NAME
      valueFrom:
        fieldRef:
           fieldPath: spec.nodeName
    - name: KUBERNETES_LIMIT_CPU
      valueFrom:
        resourceFieldRef:
           containerName: ${POD_NAME}-c1
           resource: limits.cpu
    - name: KUBERNETES_LIMIT_MEM
      valueFrom:
        resourceFieldRef:
           containerName: ${POD_NAME}-c1
           resource: limits.memory
    - name: MEM_LIMIT
      valueFrom:
        resourceFieldRef:
           containerName: ${POD_NAME}-c1
           resource: limits.memory
    image: ${K8S_DOCKER_IMAGE}
    imagePullPolicy: IfNotPresent
    tty: true
    stdin: true
    command: [ "$K8S_INIT", "--", "$K8S_INITENV", "$K8S_ENTRYPOINT", $K8S_ENTRYPOINT_ARGS_EXPANDED ]
    workingDir: ${K8S_HOME_DIR_MOUNT}
    name: ${POD_NAME}-c1
    ports:
    - containerPort: ${PROXY_PORT}
      protocol: TCP
    resources:
      requests:
        cpu: ${K8S_NUM_CPU_REQ}
        memory: ${K8S_GB_MEM}Gi
        ${K8S_GPU_RESOURCE_TYPE}: ${K8S_NUM_GPU}
      limits:
        cpu: ${K8S_NUM_CPU}
        memory: ${K8S_GB_MEM}Gi
        ${K8S_GPU_RESOURCE_TYPE}: ${K8S_NUM_GPU}
    volumeMounts:
    #- mountPath: ${K8S_HOME_DIR_MOUNT}
      #name: ${K8S_USERNAME}-home
${NBGRADER_MOUNTS}
${CUDA_LIB_MOUNTS}
    - mountPath: /opt/k8s-support
      name: support
    - mountPath: /datasets
      name: dsmlp-datasets
    - mountPath: /dev/shm
      name: dshm
  restartPolicy: Never
  terminationGracePeriodSeconds: 30
  activeDeadlineSeconds:  ${K8S_TIMEOUT_SECONDS}
  volumes:
${NBGRADER_VOLS}
${CUDA_LIB_VOLS}
  - name: support  
    emptyDir: {}
  - name: dsmlp-datasets  
    persistentVolumeClaim:
      claimName:  ${K8S_DATASETS_PVC}
  - name: dshm
    emptyDir:
      medium:  Memory
EOM

}

function create_pod() {
	create_pod_spec > spec.yml
	create_pod_spec | kubectl create -f - 1>&3
}

findport() {

	declare -A PORTS_IN_USE
	for p in $( netstat -a -t -n | grep LISTEN | while read -a INP; do
		IFS=\: read HOST PORT <<<"${INP[3]}"
		if [ "${INP[5]}" = "LISTEN" ]; then
			echo $PORT
		fi
	done ); do
		PORTS_IN_USE[$p]=1
	done

	TGT=""
	for ((i= $(( 8000 + ($K8S_UID % 12500) )) ;i<=25000;i++)); do 
		if [[ ! ${PORTS_IN_USE[$i]} ]] ; then		
			TGT=$i
			break
		fi
	done

	echo $TGT
}

syslog() {
	LOGPREFIX="launch-sh"
	LOGHOST=acs-syslog.ucsd.edu
	LOGPORT=514
	LOGLEVEL=local4.info
 	LOGEXTRA=  # --rfc3164
	echo "$@" | logger -t ${LOGPREFIX} -d -n ${LOGHOST} -P ${LOGPORT} -p ${LOGLEVEL} ${LOGEXTRA}
}

main "$@"

