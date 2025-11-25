"""
Comprehensive skills database for resume parsing and job matching.
Contains technical skills, programming languages, frameworks, tools, and soft skills.
"""

# Programming Languages
PROGRAMMING_LANGUAGES = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'golang',
    'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    'shell', 'bash', 'powershell', 'sql', 'pl/sql', 't-sql', 'html', 'css',
    'sass', 'scss', 'less', 'dart', 'lua', 'erlang', 'elixir', 'haskell',
    'clojure', 'f#', 'vb.net', 'objective-c', 'assembly', 'fortran', 'cobol'
]

# Data Science & Machine Learning
DATA_SCIENCE_ML = [
    'machine learning', 'deep learning', 'neural networks', 'tensorflow', 'pytorch',
    'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
    'plotly', 'jupyter', 'r studio', 'data science', 'data analysis', 'statistics',
    'regression', 'classification', 'clustering', 'nlp', 'natural language processing',
    'computer vision', 'opencv', 'xgboost', 'lightgbm', 'catboost', 'spark ml',
    'mlflow', 'kubeflow', 'tensorboard', 'weka', 'rapids', 'dask'
]

# Big Data & Data Engineering
BIG_DATA = [
    'hadoop', 'spark', 'kafka', 'flink', 'storm', 'samza', 'airflow', 'luigi',
    'prefect', 'dagster', 'dbt', 'snowflake', 'bigquery', 'redshift', 'databricks',
    'hive', 'pig', 'hbase', 'cassandra', 'mongodb', 'elasticsearch', 'solr',
    'neo4j', 'redis', 'memcached', 'dynamodb', 'cosmos db', 'couchdb', 'riak'
]

# Cloud Platforms
CLOUD_PLATFORMS = [
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'oracle cloud',
    'ibm cloud', 'alibaba cloud', 'digitalocean', 'heroku', 'vercel', 'netlify',
    'cloudflare', 'linode', 'vultr', 'rackspace'
]

# AWS Services
AWS_SERVICES = [
    'ec2', 's3', 'lambda', 'rds', 'dynamodb', 'redshift', 'emr', 'glue', 'athena',
    'kinesis', 'sqs', 'sns', 'api gateway', 'cloudformation', 'cloudwatch',
    'iam', 'vpc', 'route53', 'elb', 'alb', 'nlb', 'ecs', 'eks', 'fargate',
    'ecr', 'codecommit', 'codebuild', 'codedeploy', 'codepipeline', 'sagemaker',
    'rekognition', 'comprehend', 'translate', 'polly', 'lex', 'connect'
]

# Azure Services
AZURE_SERVICES = [
    'azure functions', 'azure storage', 'azure sql', 'cosmos db', 'azure data factory',
    'azure databricks', 'azure synapse', 'azure ml', 'azure cognitive services',
    'azure devops', 'azure pipelines', 'azure kubernetes service', 'aks',
    'azure container instances', 'azure app service', 'azure key vault'
]

# GCP Services
GCP_SERVICES = [
    'compute engine', 'cloud storage', 'bigquery', 'cloud sql', 'cloud spanner',
    'dataflow', 'dataproc', 'bigtable', 'cloud functions', 'cloud run',
    'kubernetes engine', 'gke', 'cloud build', 'cloud deploy', 'ai platform',
    'vertex ai', 'cloud ml', 'cloud vision', 'cloud nlp', 'cloud speech'
]

# DevOps & CI/CD
DEVOPS = [
    'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 'github actions',
    'circleci', 'travis ci', 'bamboo', 'teamcity', 'azure devops', 'terraform',
    'ansible', 'puppet', 'chef', 'saltstack', 'vagrant', 'packer', 'consul',
    'vault', 'nomad', 'prometheus', 'grafana', 'elk stack', 'elastic stack',
    'splunk', 'datadog', 'new relic', 'appdynamics', 'sentry', 'rollbar'
]

# Web Frameworks
WEB_FRAMEWORKS = [
    'react', 'angular', 'vue', 'vue.js', 'ember', 'backbone', 'svelte',
    'next.js', 'nuxt.js', 'gatsby', 'remix', 'django', 'flask', 'fastapi',
    'express', 'koa', 'nest.js', 'spring', 'spring boot', 'asp.net', 'laravel',
    'symfony', 'ruby on rails', 'rails', 'phoenix', 'play framework', 'gin',
    'echo', 'fiber', 'gin framework'
]

# Mobile Development
MOBILE = [
    'ios', 'android', 'react native', 'flutter', 'xamarin', 'ionic', 'cordova',
    'swift', 'swiftui', 'kotlin', 'java android', 'objective-c', 'xcode',
    'android studio', 'app development', 'mobile development'
]

# Databases
DATABASES = [
    'mysql', 'postgresql', 'postgres', 'oracle', 'sql server', 'sqlite',
    'mongodb', 'cassandra', 'redis', 'elasticsearch', 'dynamodb', 'neo4j',
    'couchdb', 'riak', 'influxdb', 'timescaledb', 'cockroachdb', 'mariadb',
    'aurora', 'rds', 'cosmos db', 'firebase', 'firestore', 'realm'
]

# Testing
TESTING = [
    'unit testing', 'integration testing', 'e2e testing', 'test automation',
    'selenium', 'cypress', 'playwright', 'puppeteer', 'jest', 'mocha', 'chai',
    'jasmine', 'pytest', 'unittest', 'junit', 'testng', 'rspec', 'cucumber',
    'gherkin', 'bdd', 'tdd', 'qa', 'quality assurance', 'manual testing'
]

# Version Control & Collaboration
VERSION_CONTROL = [
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'perforce',
    'jira', 'confluence', 'trello', 'asana', 'monday.com', 'slack', 'microsoft teams',
    'zoom', 'webex', 'agile', 'scrum', 'kanban', 'sprint planning'
]

# Monitoring & Logging
MONITORING = [
    'prometheus', 'grafana', 'datadog', 'new relic', 'appdynamics', 'splunk',
    'elk', 'elasticsearch', 'logstash', 'kibana', 'fluentd', 'fluentbit',
    'cloudwatch', 'azure monitor', 'stackdriver', 'sentry', 'rollbar', 'bugsnag'
]

# Security
SECURITY = [
    'cybersecurity', 'penetration testing', 'vulnerability assessment', 'owasp',
    'ssl', 'tls', 'oauth', 'oauth2', 'jwt', 'saml', 'ldap', 'active directory',
    'encryption', 'hashing', 'cryptography', 'firewall', 'vpn', 'siem', 'soc',
    'security audit', 'compliance', 'gdpr', 'hipaa', 'pci dss', 'iso 27001'
]

# Business Intelligence & Analytics
BI_ANALYTICS = [
    'tableau', 'power bi', 'qlik', 'qlikview', 'qliksense', 'looker', 'metabase',
    'superset', 'apache superset', 'microstrategy', 'sas', 'spss', 'stata',
    'business intelligence', 'data visualization', 'dashboard', 'reporting',
    'etl', 'elt', 'data warehousing', 'olap', 'data modeling', 'star schema',
    'snowflake schema'
]

# Software Engineering Practices
SOFTWARE_PRACTICES = [
    'agile', 'scrum', 'kanban', 'lean', 'devops', 'ci/cd', 'continuous integration',
    'continuous deployment', 'microservices', 'rest api', 'graphql', 'soap',
    'api design', 'system design', 'architecture', 'design patterns', 'solid',
    'clean code', 'code review', 'pair programming', 'tdd', 'bdd', 'refactoring'
]

# Soft Skills
SOFT_SKILLS = [
    'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving',
    'critical thinking', 'analytical thinking', 'creativity', 'adaptability',
    'time management', 'project management', 'stakeholder management',
    'client relations', 'customer service', 'presentation', 'public speaking',
    'mentoring', 'coaching', 'training', 'documentation', 'technical writing'
]

# Combine all technical skills
ALL_TECHNICAL_SKILLS = (
    PROGRAMMING_LANGUAGES +
    DATA_SCIENCE_ML +
    BIG_DATA +
    CLOUD_PLATFORMS +
    AWS_SERVICES +
    AZURE_SERVICES +
    GCP_SERVICES +
    DEVOPS +
    WEB_FRAMEWORKS +
    MOBILE +
    DATABASES +
    TESTING +
    VERSION_CONTROL +
    MONITORING +
    SECURITY +
    BI_ANALYTICS +
    SOFTWARE_PRACTICES
)

# All skills (technical + soft)
ALL_SKILLS = ALL_TECHNICAL_SKILLS + SOFT_SKILLS

# Create a normalized mapping (lowercase for matching)
SKILLS_DICT = {skill.lower(): skill for skill in ALL_SKILLS}

# Skills by category for better organization
SKILLS_BY_CATEGORY = {
    'programming_languages': PROGRAMMING_LANGUAGES,
    'data_science_ml': DATA_SCIENCE_ML,
    'big_data': BIG_DATA,
    'cloud': CLOUD_PLATFORMS + AWS_SERVICES + AZURE_SERVICES + GCP_SERVICES,
    'devops': DEVOPS,
    'web_frameworks': WEB_FRAMEWORKS,
    'mobile': MOBILE,
    'databases': DATABASES,
    'testing': TESTING,
    'version_control': VERSION_CONTROL,
    'monitoring': MONITORING,
    'security': SECURITY,
    'bi_analytics': BI_ANALYTICS,
    'software_practices': SOFTWARE_PRACTICES,
    'soft_skills': SOFT_SKILLS
}


def extract_skills_from_text(text: str, min_confidence: float = 0.7) -> list:
    """
    Extract skills from text using the comprehensive skills database.
    
    Args:
        text: Text to extract skills from
        min_confidence: Minimum confidence threshold (0.0-1.0)
    
    Returns:
        List of extracted skills (normalized)
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_skills = []
    
    # Direct matches (exact or word boundary)
    for skill_key, skill_name in SKILLS_DICT.items():
        # Check for exact word match (word boundaries)
        import re
        pattern = r'\b' + re.escape(skill_key) + r'\b'
        if re.search(pattern, text_lower):
            if skill_name not in found_skills:
                found_skills.append(skill_name)
    
    # Also check for partial matches for multi-word skills
    # (e.g., "machine learning" should match even if not exact word boundaries)
    for skill_key, skill_name in SKILLS_DICT.items():
        if ' ' in skill_key:  # Multi-word skill
            if skill_key in text_lower and skill_name not in found_skills:
                found_skills.append(skill_name)
    
    return found_skills


def categorize_skills(skills: list) -> dict:
    """
    Categorize extracted skills by type.
    
    Args:
        skills: List of skill names
    
    Returns:
        Dictionary mapping categories to skills
    """
    categorized = {category: [] for category in SKILLS_BY_CATEGORY.keys()}
    categorized['other'] = []
    
    skills_lower = [s.lower() for s in skills]
    
    for category, category_skills in SKILLS_BY_CATEGORY.items():
        category_skills_lower = [s.lower() for s in category_skills]
        for skill in skills:
            if skill.lower() in category_skills_lower:
                categorized[category].append(skill)
    
    # Find uncategorized skills
    categorized_lower = set()
    for skills_list in categorized.values():
        categorized_lower.update(s.lower() for s in skills_list)
    
    for skill in skills:
        if skill.lower() not in categorized_lower:
            categorized['other'].append(skill)
    
    return categorized


def get_skill_synonyms(skill: str) -> list:
    """
    Get synonyms or alternative names for a skill.
    
    Args:
        skill: Skill name
    
    Returns:
        List of synonyms
    """
    synonyms_map = {
        'python': ['python3', 'python 3'],
        'javascript': ['js', 'ecmascript'],
        'typescript': ['ts'],
        'c++': ['cpp', 'cplusplus'],
        'c#': ['csharp', 'c sharp'],
        'react': ['reactjs', 'react.js'],
        'vue': ['vuejs', 'vue.js'],
        'angular': ['angularjs', 'angular.js'],
        'postgresql': ['postgres'],
        'machine learning': ['ml', 'machine-learning'],
        'deep learning': ['dl', 'deep-learning'],
        'natural language processing': ['nlp'],
        'artificial intelligence': ['ai'],
        'data science': ['data-science'],
        'big data': ['big-data'],
        'aws': ['amazon web services'],
        'gcp': ['google cloud', 'google cloud platform'],
        'kubernetes': ['k8s'],
        'docker': ['docker container'],
        'ci/cd': ['cicd', 'continuous integration', 'continuous deployment'],
        'rest api': ['rest', 'restful api'],
        'api': ['application programming interface'],
    }
    
    skill_lower = skill.lower()
    if skill_lower in synonyms_map:
        return synonyms_map[skill_lower]
    return []

