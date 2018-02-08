#!/bin/sh
echo $HOSTNAME
## MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/mtang73/Disney

CAFFE_DIR=${ROOT_DIR}/deeplab/code
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=pascal_box

NUM_LABELS=21
DATA_ROOT=${ROOT_DIR}/data/pascal_box/

## Specify which model to train
########### voc12 ################
NET_ID=deeplab_largeFOV
#NET_ID=deeplab_vgg16
#NET_ID=deeplab_msc_largeFOV
#NET_ID=resnet-101


## Variables used for weakly or semi-supervisedly training
TRAIN_SET_SUFFIX=
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

RUN_TRAIN=0
RUN_TEST=1

TEMP="$1"

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
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
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo $CMD
		echo Running ${CMD} && ${CMD}
		echo $CMD
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
		TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
		#TEST_ITER=100
        #MODEL=${EXP}/model/${NET_ID}/test.caffemodel
        MODEL=${EXP}/model/${NET_ID}/train_iter_$TEMP.caffemodel
		if [ ! -f ${MODEL} ]; then
			MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
		fi
		#
		echo Testing net ${EXP}/${NET_ID}
		FEATURE_DIR=${EXP}/features/${NET_ID}
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
		sed "$(eval echo $(cat sub.sed))" \
			${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
		CMD="${CAFFE_BIN} test \
           --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
           --weights=${MODEL} \
           --gpu=${DEV_ID} \
           --iterations=${TEST_ITER}"
		echo $CMD
		echo Running ${CMD} && ${CMD}
		echo $CMD
    done
fi


