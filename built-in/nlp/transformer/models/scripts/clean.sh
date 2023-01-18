CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))

rm -rf ${EXAMPLES_DIR}/models &> /dev/null
rm -rf ${EXAMPLES_DIR}/logs &> /dev/null
rm -rf ${EXAMPLES_DIR}/__pycache__ &> /dev/null
rm -rf ${EXAMPLES_DIR}/cnnl_auto_log &> /dev/null
rm -rf ${EXAMPLES_DIR}/models &> /dev/null
