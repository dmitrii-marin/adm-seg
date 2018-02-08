#!/bin/sh

## MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/mtang73/Disney

CAFFE_DIR=${ROOT_DIR}/deeplab/code
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

THISHOST=$(hostname)

EXP=msra10k
if [ "${EXP}" = "msra10k" ]; then
    NUM_LABELS=2
    if [ "${THISHOST}" = "DRZ-HAL" ]; then
        DATA_ROOT=${ROOT_DIR}/data/MSRA10K/
    elif [ "${THISHOST}" = "hal.rndr.csd.uwo.ca" ]; then
        DATA_ROOT=/data/mtang73/dataset/MSRA10K/
    else
        return
    fi
elif [ "${EXP}" = "voc12" ]; then
    NUM_LABELS=2
    if [ "${THISHOST}" = "DRZ-HAL" ]; then
        DATA_ROOT=${ROOT_DIR}/data/VOCdevkit/VOC2012/
    elif [ "${THISHOST}" = "hal.rndr.csd.uwo.ca" ]; then
        DATA_ROOT=/data/mtang73/dataset/VOC2012/
    else
        return
    fi
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
#NET_ID=deeplab_largeFOV
#NET_ID=deeplab_RESNET101
NET_ID=deeplab_vgg16


## Variables used for weakly or semi-supervisedly training
#TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug

#TRAIN_SET_STRONG=train
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

DEV_ID=0

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAINWITHNCLOSS="$1"

## Training #1 (on train_aug)

if [ ${RUN_TRAINWITHNCLOSS} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in trainwithncloss solverwithncloss; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solverwithncloss_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo Running ${CMD} && ${CMD}
fi

