FROM python:3.7

WORKDIR /usr/src/app

RUN pip install pipenv

COPY . .

RUN pipenv install --dev --ignore-pipfile --deploy --system --clear

CMD [ "pipenv", "run", "voila", "--port", "12345", "--no-browser", "--template=vuetify-default", "--MappingKernelManager.cull_interval=10", "--MappingKernelManager.cull_idle_timeout=10", "interactive-service.ipynb" ]
