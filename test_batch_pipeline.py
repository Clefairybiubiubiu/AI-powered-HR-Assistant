import requests
import json

BASE = "http://127.0.0.1:8000/api/v1"

jobs = [
    {
        "job_title": "Health Records Analyst",
        "job_description": """Job Summary:
The Health Information Management (HIM) Health Records Analyst manages patient data within the healthcare facility, ensuring the accuracy, organization, and security of health records. Additionally, this position provides reasonable assurance to the organization that patient care and satisfaction remain a priority within the day-to-day operations. This is achieved by compiling, processing, maintaining, and sustaining medical records of hospital and clinic patients in a manner consistent with medical, administrative, ethical, legal, and regulatory requirements of the healthcare system.

Job Duties:
- Learns the fundamentals of health records and health information management (HIM)
- Develops a general understanding of Epic Chart Desk and related HIM systems
- Processes health records (scanning, retention, adoption cases, date of death processing, etc.)
- Develops an understanding of policies/procedures and scanning methodologies (FIMS, Clinic Scanning)
- Answers phone calls, satisfies requests, and assists with chart corrections
- Works in an office environment ensuring compliance with all organizational policies

Education: High School Diploma or Equivalent (GED) required
Values: Kindness, Excellence, Learning, Innovation, Safety
"""
    },
    {
        "job_title": "HRIT Application Development Support Analyst",
        "job_description": """About the job:
We're seeking an experienced HRIT Application Development Support Analyst who will play a pivotal role in maintaining and evolving our HR technology landscape. This position supports and enhances cloud-based HR platforms such as Workday and other SaaS applications.

Responsibilities:
- Provide hands-on technical support across HR systems
- Monitor performance, troubleshoot, and ensure system stability
- Support HR technology projects including enhancements and upgrades
- Assist with data integrity, imports/exports, and compliance
- Collaborate with HR, IT, and vendors for continuous improvement
- Identify automation opportunities and implement process improvements
- Ensure adherence to GDPR, HIPAA, and corporate data security policies

Requirements:
- 3+ years in HRIT or system analyst roles
- Hands-on experience with Workday or similar HR applications
- Technical familiarity with SQL, Power BI, or Tableau
- Bachelor's degree in Information Systems, HR, or related field
"""
    },
    {
        "job_title": "AI Operations Analyst",
        "job_description": """About the job:
Southwind is a leading innovator in the home services industry with AI-enabled contact centers live across dozens of markets. This role focuses on ensuring AI-driven customer interactions are accurate, efficient, and continuously improving.

Key Responsibilities:
- Monitor AI-initiated calls for accuracy and escalation handling
- Perform QA audits, workflow reviews, and training
- Analyze call data, identify misclassifications, and propose fixes
- Maintain data hygiene, track KPIs (contact rate, conversion, cancellation)
- Collaborate with CRO, AI ops, and third-party vendors to optimize prompts and routing
- Document and communicate operational workflows and best practices

Requirements:
- Bachelor's Degree required
- 1+ years experience in a professional, data-driven environment
- Strong attention to detail, QA mindset, and spreadsheet skills
- Familiarity with Salesforce, Google Workspace, or telephony platforms
"""
    }
]

job_ids = []
for job in jobs:
    res = requests.post(f"{BASE}/jobs/analyze", json=job)
    job_data = res.json()
    job_ids.append(job_data.get("job_id"))
    print(f"✅ Uploaded job: {job_data.get('job_title')} (ID: {job_data.get('job_id')})")

resume_files = [
    "/Users/junfeibai/Desktop/Junfei Bai - CV.pdf",
    "/Users/junfeibai/Desktop/Junfei Bai CV.pdf"
]
  # change to your own files

candidate_ids = []
for file_name in resume_files:
    with open(file_name, "rb") as f:
        files = {"file": f}
        res = requests.post(f"{BASE}/candidates/parse", files=files)
        cand_data = res.json()
        candidate_ids.append(cand_data.get("candidate_id"))
        print(f"✅ Uploaded candidate: {cand_data.get('name', 'Unknown')} (ID: {cand_data.get('candidate_id')})")

payload = {"job_ids": job_ids, "candidate_ids": candidate_ids}
res = requests.post(f"{BASE}/scoring/batch-scoring", json=payload)
print("\n🎯 Batch Scoring Results:")
print(json.dumps(res.json(), indent=2))

for job_id in job_ids:
    res = requests.get(f"{BASE}/scoring/scores/{job_id}")
    print(f"\n📊 Scores for job {job_id}:")
    print(json.dumps(res.json(), indent=2))
