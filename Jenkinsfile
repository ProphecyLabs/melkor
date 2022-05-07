pipeline {
    agent any

    stages {
        stage('Hello') {
            steps {
                echo 'Hello Melkor'
            }
        }
        stage('Test_Clean') {
            steps {
                sh 'tests/run_test.bat'
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
