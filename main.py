from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

def extract_resume_details(resume_text):
    resume_extraction_template = """
    You are a professional resume analyzer. Your task is to extract relevant details from the resume provided below.

    Resume:
    {resume}

    Extract the following:
    1. Candidate's name:
    2. Contact information:
    3. Education:
    4. Work experience (Company, Role, Duration):
    5. Key skills:
    6. Certifications:
    7. Summary of experience:
    """
    resume_extraction_prompt = PromptTemplate(
        input_variables=["resume"],
        template=resume_extraction_template,
    )
    resume_extraction_chain = LLMChain(llm=llm, prompt=resume_extraction_prompt)
    return resume_extraction_chain.run(resume=resume_text)

def match_resume_with_job(resume_details, job_description):
    job_match_template = """
    You are a professional resume analyzer. Below is the candidate's resume data and a job description.
    Please analyze how well the candidate matches the job requirements.

    Candidate's Details:
    {resume_details}

    Job Description:
    {job_description}

    Provide an analysis with the following:
    1. How well does the candidate's experience match the job's required experience?
    2. How relevant are the candidate's skills to the job?
    3. Any key areas where the candidate might fall short?
    4. A score between 1-10 on the candidate's overall fit for this role.
    """
    job_match_prompt = PromptTemplate(
        input_variables=["resume_details", "job_description"],
        template=job_match_template,
    )
    job_match_chain = LLMChain(llm=llm, prompt=job_match_prompt)
    return job_match_chain.run(resume_details=resume_details, job_description=job_description)

def analyze_resume(resume_text, job_description_texts):
    resume_details = extract_resume_details(resume_text)
    match_analysis = match_resume_with_job(resume_details, job_description_texts)
    return {
        "resume_details": resume_details,
        "match_analysis": match_analysis
    }

resume_texts = """
    John Doe
    Email: johndoe@example.com
    Phone: (555) 555-5555
    
    Education:
    - Bachelor of Science in Computer Science, XYZ University, 2018
    
    Work Experience:
    - Software Engineer, ABC Company (Jan 2019 - Present)
      - Developed full-stack applications using Python and JavaScript.
      - Led a team of 5 engineers to develop and maintain enterprise-grade software.
      - Integrated REST APIs for internal tools and third-party services.
    
    - Junior Developer, DEF Corporation (Jun 2018 - Dec 2018)
      - Assisted in building web applications using React and Node.js.
    
    Skills:
    - Python, JavaScript, React, Node.js, SQL, Docker, Git
    - Leadership, Agile methodologies
    
    Certifications:
    - AWS Certified Solutions Architect
    
    Summary:
    Highly motivated software engineer with 3+ years of experience in full-stack development. Passionate about building scalable and efficient software solutions.
    """
job_description_text = """
    We are seeking an experienced software engineer with the following requirements:
    - 3+ years of experience in software development.
    - Expertise in Python, JavaScript, and modern web frameworks (React, Node.js).
    - Strong understanding of cloud services (AWS preferred).
    - Experience with SQL databases and containerization (Docker).
    - Excellent team collaboration and leadership skills.
    - Preferred: AWS certification.
    """
result = analyze_resume(resume_texts, job_description_text)

print("Extracted Resume Details:")
print(result["resume_details"])
print("\nJob Match Analysis:")
print(result["match_analysis"])
