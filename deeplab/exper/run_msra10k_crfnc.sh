#!/bin/sh
STARTTIME=$(date +%s)
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
RUN_TEST="$1"
RUN_TESTCRF="$2"
RUN_TESTNCLAYER="$3" # replicated layers
RUN_TESTNCMULTISTAGE="$4"
RUN_TRAINNCMULTISTAGE="$5"

## baseline

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
				#if [ ! -f ${MODEL} ]; then
				#		MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				#fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8softmax
                mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

## CRF

if [ ${RUN_TESTCRF} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/train_iter_100.caffemodel
				TEST_ITER=1
				#if [ ! -f ${MODEL} ]; then
				#		MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				#fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8softmax
                mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/testcrf.prototxt > ${CONFIG_DIR}/testcrf_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/testcrf_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

## NC layers
#BI_WEIGHT=200000
BI_WEIGHT=20000
BI_XY_STD=50
BI_RGB_STD=15
    

if [ ${RUN_TESTNCLAYER} -eq 1 ]; then
    #
    ELTWISE_SCALING=2e-8
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				TEST_ITER=100
				MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
				#if [ ! -f ${MODEL} ]; then
				#		MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				#fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8softmax
                mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/testnclayer.prototxt > ${CONFIG_DIR}/testnclayer_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/testnclayer_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

if [ ${RUN_TESTNCMULTISTAGE} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				#MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
				MODEL=${EXP}/model/${NET_ID}/trainncmultistage_iter_4000.caffemodel
				TEST_ITER=1000
				#if [ ! -f ${MODEL} ]; then
				#		MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				#fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8softmax
                mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/testncmultistage.prototxt > ${CONFIG_DIR}/testncmultistage_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/testncmultistage_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

if [ ${RUN_TRAINNCMULTISTAGE} -eq 1 ]; then
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
    MODEL=${EXP}/model/${NET_ID}/train_iter_2000.caffemodel
    #MODEL=${EXP}/model/${NET_ID}/init.caffemodel
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in trainncmultistage solverncmultistage; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solverncmultistage_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo ${CMD}
		echo Running ${CMD} && ${CMD}
fi


ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task..."
