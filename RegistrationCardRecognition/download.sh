#!/bin/bash
optspec=":hv-:"
while getopts "$optspec" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                category)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    CATEGORY=$val
                    ;;
                category=*)
                    val=${OPTARG#*=}
                    opt=${OPTARG%=$val}
                    CATEGORY=$val
                    ;;
                type)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    TYPE=$val
                    ;;
                type=*)
                    val=${OPTARG#*=}
                    opt=${OPTARG%=$val}
                    TYPE=$val
                    ;;    
                model)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    MODEL=$val
                    ;;
                model=*)
                    val=${OPTARG#*=}
                    opt=${OPTARG%=$val}
                    MODEL=$val
                    ;;
                *)
                    if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" = ":" ]; then
                        echo "Unknown option --${OPTARG}" >&2
                    fi
                    ;;
            esac;;
        h)
            echo "usage: $0 [-v] [--loglevel[=]<value>]" >&2
            exit 2
            ;;
        v)
            echo "Parsing option: '-${optchar}'" >&2
            ;;
        *)
            if [ "$OPTERR" != 1 ] || [ "${optspec:0:1}" = ":" ]; then
                echo "Non-option argument: '-${OPTARG}'" >&2
            fi
            ;;
    esac
done
if [ "$CATEGORY" == "" ]; then 
    echo "--category is missing" 
fi
if [ "$TYPE" == "" ]; then 
    echo "--type is missing" 
fi
if [ "$MODEL" == "" ]; then 
    echo "--model is missing" 
fi
mkdir -p data/$CATEGORY/train/$TYPE/$MODEL;
scp -i ./data/data-science.pem -r \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/data/$CATEGORY/train/$TYPE/$MODEL/* \
data/$CATEGORY/train/$TYPE/$MODEL;