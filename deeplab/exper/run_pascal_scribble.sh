#!/bin/sh

HOSTNAME=$(hostname)

echo $HOSTNAME

echo Setting up for HAL
ROOT_DIR=..
CAFFE_DIR=${ROOT_DIR}/code
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe
## MODIFY PATH for YOUR SETTING
DATA_ROOT=/home/mtang73/Disney/data/pascal_scribble/

export PYTHONPATH=../code/python:$PYTHONPATH

if [ -z "$EXP" ]; then EXP=pascal_scribble; fi

NUM_LABELS=21

## Specify which model to train

if [ -z "$NET_ID" ]; then
NET_ID=deeplab_largeFOV
#NET_ID=deeplab_vgg16
#NET_ID=resnet-101
#NET_ID=deeplab_msc_largeFOV
fi
if [ -z "$DEV_ID" ]; then DEV_ID=0; fi

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run
# train with partial CE only
if [ -z "$RUN_TRAIN" ]; then RUN_TRAIN=1; fi
if [ -z "$RUN_TEST"  ]; then RUN_TEST=1;  fi
# finetune with CUT loss (denseCRF)

TEMP="$1"

set -e

if [ ${RUN_TRAIN} -eq 1 ]; then
    case "$TRAIN_CONFIG" in
	DENSE-GD)
		TRAIN_CONFIG="trainwithcutloss solverwithcutloss"
		;;
	DENSE-ADM)
		TRAIN_CONFIG="trainlampert solverlampert"
		;;
  	GRID-GD)
		TRAIN_CONFIG="trainwithsparcecutloss solverwithsparcecutloss"
		[ -n "$CUT_LOSS_WEIGHT" ] || CUT_LOSS_WEIGHT=3.5
		;;
	GRID-ADM)
		TRAIN_CONFIG="trainwithgc solverwithgc"
		[ -n "$GC_ITERATION" ] || GC_ITERATION=5
		[ -n "$GC_POTTS" ] || GC_POTTS=80
		[ -n "$GC_CEWEIGHT" ] || GC_CEWEIGHT=1.5
		;;
	"")
		TRAIN_CONFIG="train solver"
		;;
	*)
		echo Unknown TRAIN_CONFIG=$TRAIN_CONFIG
		exit 1
    esac
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    #
    if [ -z "$MODEL" ]; then MODEL=${EXP}/model/${NET_ID}/vgg16_20M.caffemodel; fi
    echo Training net ${EXP}/${NET_ID}
    echo "$(eval echo $(cat sub.sed))" > sub_impl.sed
    for pname in $TRAIN_CONFIG; do
	sed -f sub_impl.sed ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
	SOLVER=$pname
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/${SOLVER}_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID} \
         --weights=${MODEL}"
    echo $CMD
    echo Running ${CMD} && ${CMD}
    echo $CMD

    # reset model variable
    MODEL=
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    if [ -z "$TEST_SETS" ]; then TEST_SETS=val; fi
    for TEST_SET in $TEST_SETS; do
	TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
	MODEL=${EXP}/model/${NET_ID}/${TEMP}.caffemodel
	echo $MODEL
	if [ ! -f ${MODEL} ]; then
		MODEL=$(ls -t ${EXP}/model/${NET_ID}/*.caffemodel | head -n 1)
	fi
	echo Testing model $MODEL
	echo Testing net ${EXP}/${NET_ID}
	FEATURE_DIR=${EXP}/features/${NET_ID}
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
        echo "$(eval echo $(cat sub.sed))" > sub_impl.sed
	sed -f sub_impl.sed ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
		--model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
		--weights=${MODEL} \
		--gpu=${DEV_ID} \
		--iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
	echo $CMD
    done
fi

