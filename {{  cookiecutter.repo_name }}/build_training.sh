STAGE=training

PROJECT_DIR="$( cd "$( dirname "$0}" )" >/dev/null 2>&1 && pwd )"
cd "${PROJECT_DIR}" || exit;

AWS_ACCESS_KEY_ID=$(aws --profile {{ cookiecutter.aws_profile }} configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws --profile {{ cookiecutter.aws_profile }} configure get aws_secret_access_key)

PROJECT_VERSION="$(git rev-parse HEAD)"

source ${PROJECT_DIR}/env

docker build -t {{cookiecutter.project_name}}:$PROJECT_VERSION . \
    --build-arg PIP_INDEX=$PIP_INDEX \
    --build-arg PIP_HOST=$PIP_HOST \
    --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY 
