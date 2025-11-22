pipeline {
    agent any
    
    environment {
        PROJECT_ID = credentials('gcp-project-id')
        REGION = 'us-central1'
        GCP_CREDENTIALS = credentials('gcp-service-account')
        MODELKIT_NAME = 'sentiment-classifier'
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    sh '''
                        curl -L https://github.com/kitops-ml/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz | tar -xz
                        sudo mv kit /usr/local/bin/
                        
                        gcloud auth activate-service-account --key-file=${GCP_CREDENTIALS}
                        gcloud config set project ${PROJECT_ID}
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev
                    '''
                }
            }
        }
        
        stage('Build ModelKit') {
            steps {
                script {
                    def version = readFile('version.txt').trim()
                    env.MODELKIT_URI = "${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/${MODELKIT_NAME}:${version}"
                    
                    sh """
                        kit pack . -t ${MODELKIT_URI}
                        kit push ${MODELKIT_URI}
                    """
                }
            }
        }
        
        stage('Validate ModelKit') {
            steps {
                sh '''
                    python -m pip install pytest pykitops
                    kit pull ${MODELKIT_URI}
                    kit unpack ${MODELKIT_URI} -d ./validation
                    pytest tests/validation/ -v
                '''
            }
        }
        
        stage('Deploy Pipeline') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    python -m pip install google-cloud-aiplatform kfp
                    python scripts/trigger_vertex_pipeline.py \
                        --project-id ${PROJECT_ID} \
                        --region ${REGION} \
                        --modelkit-uri ${MODELKIT_URI}
                '''
            }
        }
        
        stage('Monitor Pipeline') {
            steps {
                sh '''
                    python scripts/monitor_pipeline.py \
                        --project-id ${PROJECT_ID} \
                        --region ${REGION} \
                        --timeout 3600
                '''
            }
        }
    }
    
    post {
        success {
            echo "Pipeline completed successfully!"
            // Send notifications
        }
        failure {
            echo "Pipeline failed!"
            // Send alerts
        }
    }
}
