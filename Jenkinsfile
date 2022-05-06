pipeline {
    agent any

    stages {
        stage('Hello') {
            steps {
                echo 'Hello Melkor'
            }
        }
        stage('CheckWd') {
            steps {
                sh 'pwd'
            }
        }
        stage('Build') {
            steps {
                echo 'Building'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying'
            }
        }
        stage('Release') {
            steps {
                echo 'Release'
            }
        }
    }
}
