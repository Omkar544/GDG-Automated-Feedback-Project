from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
import os
import datetime
import cv2
import numpy as np
import re
import psycopg2
from werkzeug.utils import secure_filename
from joblib import load
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
import language_tool_python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.generativeai import configure, GenerativeModel
import easyocr
from PIL import Image
import tempfile

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Change this for production

# Configure Gemini AI
configure(api_key="Your Gemini key here")
gemini_model = GenerativeModel('gemini-1.5-pro-latest')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# File upload configurations
UPLOAD_FOLDER_TEACHER = 'uploads/teacher'
UPLOAD_FOLDER_STUDENT = 'uploads/student'
os.makedirs(UPLOAD_FOLDER_TEACHER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_STUDENT, exist_ok=True)

# Database configuration
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="feedback_system",
        user="postgres",
        password="Harry544",
        port="5432"
    )

# Initialize database tables
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create tables if they don't exist
        cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL,
            name VARCHAR(100) NOT NULL,
            class VARCHAR(50) NOT NULL,
            mobile_no VARCHAR(15) NOT NULL,
            email VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL,
            name VARCHAR(100) NOT NULL,
            roll_no VARCHAR(20) NOT NULL,
            class VARCHAR(50) NOT NULL,
            mobile_no VARCHAR(15) NOT NULL,
            email VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id SERIAL PRIMARY KEY,
            teacher_id INTEGER REFERENCES teachers(id),
            filename VARCHAR(255) NOT NULL,
            filepath VARCHAR(255) NOT NULL,
            class VARCHAR(50) NOT NULL,
            subject VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(id),
            assignment_id INTEGER REFERENCES assignments(id),
            filename VARCHAR(255) NOT NULL,
            filepath VARCHAR(255) NOT NULL,
            feedback TEXT,
            marks FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        cur.close()
        conn.close()

# Load ML model
try:
    model = load('scikit_model.joblib')
except FileNotFoundError:
    model = None

# Initialize grammar checker
grammar_tool = language_tool_python.LanguageTool('en-US')

# Text extraction functions
def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        return ''.join(page.extract_text() for page in reader.pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(filepath):
    try:
        doc = Document(filepath)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {str(e)}")

def extract_text_from_image(filepath):
    try:
        # First try pytesseract
        image = cv2.imread(filepath)
        text = pytesseract.image_to_string(image)
        if len(text.strip()) > 20:
            return text
            
        # Fall back to EasyOCR
        result = reader.readtext(filepath)
        return "\n".join([text for (_, text, _) in result])
    except Exception as e:
        raise RuntimeError(f"Image text extraction failed: {str(e)}")

def extract_handwritten_text(filepath):
    try:
        result = reader.readtext(filepath)
        extracted_text = "\n".join([text for (_, text, _) in result])
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise RuntimeError("Text extraction failed - low confidence")
            
        return extracted_text
    except Exception as e:
        raise RuntimeError(f"Handwritten text extraction failed: {str(e)}")

# Feedback generation
def generate_feedback_from_text(text, is_handwritten=False):
    feedback = []
    total_marks = 10
    current_marks = 10
    
    if is_handwritten:
        feedback.append("• Note: Evaluation based on handwritten content")
    else:
        if errors := grammar_tool.check(text):
            shown_errors = errors[:3]
            feedback.extend([f"• Grammar/Spelling: {error}" for error in shown_errors])
            current_marks -= len(errors) * 0.5

    try:
        prompt = f"""Evaluate this {'handwritten ' if is_handwritten else ''}assignment:
        1. Content quality (0-7)
        2. Clarity (0-2)
        3. Completeness (0-1)
        {'4. Handwriting legibility' if is_handwritten else ''}
        Provide detailed feedback and score (X/10)
        
        Student's Work:
        {text[:3000]}"""
        
        response = gemini_model.generate_content(prompt)
        ai_text = response.text
        
        if match := re.search(r'(\d+\.?\d*)/10', ai_text):
            ai_marks = float(match.group(1))
            weight = 0.6 if is_handwritten else 0.7
            current_marks = min((ai_marks * weight) + (current_marks * (1-weight)), total_marks)
        
        feedback.extend(line.strip() for line in ai_text.split('\n') 
                      if line.strip() and not line.strip().startswith('Score'))
    
    except Exception as e:
        feedback.append("• AI Feedback service unavailable")
    
    current_marks = max(0, round(current_marks, 1))
    feedback.insert(0, f"📝 Final Marks: {current_marks}/{total_marks}")
    
    return "\n".join(feedback), current_marks

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_us():
    return render_template('about.html') 
 
@app.route('/contact_us')
def contact_us():
    return render_template('contact.html')

@app.route('/teacher_auth')
def teacher_auth():
    return render_template('teacher_auth.html')

@app.route('/student_auth')
def student_auth():
    return render_template('student_auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            role = request.form['role']
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            table = 'teachers' if role == 'teacher' else 'students'
            cur.execute(f"SELECT id FROM {table} WHERE username = %s", (username,))
            if cur.fetchone():
                flash('Username already exists')
                return redirect(url_for(f'{role}_auth'))
            
            if role == 'teacher':
                cur.execute("""
                    INSERT INTO teachers (username, password, name, class, mobile_no, email)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                    username,
                    request.form['password'],
                    request.form.get('teacher_name', 'Teacher'),
                    request.form['class'],
                    request.form['mobile_no'],
                    request.form['email']
                ))
            else:
                cur.execute("""
                    INSERT INTO students (username, password, name, roll_no, class, mobile_no, email)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                    username,
                    request.form['password'],
                    request.form.get('student_name', 'Student'),
                    request.form.get('roll_no', 'N/A'),
                    request.form['class'],
                    request.form['mobile_no'],
                    request.form['email']
                ))
            
            conn.commit()
            flash('Registration successful!')
            return redirect(url_for(f'{role}_auth'))
        
        except Exception as e:
            conn.rollback()
            flash(f'Error: {str(e)}')
            return redirect(request.referrer)
        finally:
            cur.close()
            conn.close()
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        table = 'teachers' if role == 'teacher' else 'students'
        if role == 'teacher':
            cur.execute(f"""
                SELECT id, name, class FROM {table}
                WHERE username = %s AND password = %s
                """, (username, password))
        else:
            cur.execute(f"""
                SELECT id, name, class, roll_no FROM {table}
                WHERE username = %s AND password = %s
                """, (username, password))
            
        user = cur.fetchone()
        
        if user:
            session.update({
                'user_id': user[0],
                'username': username,
                'role': role,
                'name': user[1],
                'class': user[2]
            })
            if role == 'student':
                session['roll_no'] = user[3]
            
            return redirect(url_for(role))
        
        flash('Invalid credentials')
        return redirect(url_for(f'{role}_auth'))
    
    except Exception as e:
        flash(f'Login error: {str(e)}')
        return redirect(request.referrer)
    finally:
        cur.close()
        conn.close()

@app.route('/teacher')
def teacher():
    if session.get('role') != 'teacher':
        return redirect(url_for('index'))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, filename, class, subject, created_at 
            FROM assignments
            WHERE teacher_id = %s
            ORDER BY created_at DESC
            """, (session['user_id'],))
        assignments = [{
            'id': row[0],
            'filename': row[1],
            'class': row[2],
            'subject': row[3],
            'date': row[4].strftime('%Y-%m-%d %H:%M:%S')
        } for row in cur.fetchall()]
        
        cur.execute("""
            SELECT s.id, st.name, st.class, st.roll_no, a.filename, s.filename, 
                   s.created_at, s.feedback, s.marks
            FROM submissions s
            JOIN students st ON s.student_id = st.id
            JOIN assignments a ON s.assignment_id = a.id
            WHERE a.teacher_id = %s
            ORDER BY s.created_at DESC
            """, (session['user_id'],))
        
        submissions = [{
            'id': row[0],
            'name': row[1],
            'class': row[2],
            'roll_no': row[3],
            'assignment_filename': row[4],
            'filename': row[5],
            'date': row[6].strftime('%Y-%m-%d %H:%M:%S'),
            'feedback': row[7] or 'Pending',
            'marks': row[8]
        } for row in cur.fetchall()]
        
        return render_template('teacher.html',
            teacher_assignments=assignments,
            student_submissions=submissions
        )
    except Exception as e:
        flash(f'Error: {str(e)}')
        return render_template('teacher.html', teacher_assignments=[], student_submissions=[])
    finally:
        cur.close()
        conn.close()

@app.route('/student')
def student():
    if session.get('role') != 'student':
        return redirect(url_for('index'))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, filename, subject, created_at 
            FROM assignments
            WHERE class = %s
            ORDER BY created_at DESC
            """, (session['class'],))
        assignments = [{
            'id': row[0],
            'filename': row[1],
            'subject': row[2],
            'date': row[3].strftime('%Y-%m-%d %H:%M:%S')
        } for row in cur.fetchall()]
        
        cur.execute("""
            SELECT s.id, a.filename, a.subject, s.filename, s.created_at, s.feedback, s.marks
            FROM submissions s
            JOIN assignments a ON s.assignment_id = a.id
            WHERE s.student_id = %s
            ORDER BY s.created_at DESC
            """, (session['user_id'],))
        
        submissions = [{
            'id': row[0],
            'assignment_filename': row[1],
            'subject': row[2],
            'filename': row[3],
            'date': row[4].strftime('%Y-%m-%d %H:%M:%S'),
            'feedback': row[5] or 'Pending',
            'marks': row[6]
        } for row in cur.fetchall()]
        
        return render_template('student.html',
            teacher_assignments=assignments,
            student_submissions=submissions
        )
    except Exception as e:
        flash(f'Error: {str(e)}')
        return render_template('student.html', teacher_assignments=[], student_submissions=[])
    finally:
        cur.close()
        conn.close()

@app.route('/upload_teacher', methods=['POST'])
def upload_teacher():
    if session.get('role') != 'teacher':
        return redirect(url_for('index'))
    
    try:
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_TEACHER, filename)
            file.save(filepath)
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO assignments (teacher_id, filename, filepath, class, subject)
                VALUES (%s, %s, %s, %s, %s)
                """, (
                session['user_id'],
                filename,
                filepath,
                request.form['class'],
                request.form['subject']
            ))
            
            conn.commit()
            flash('Assignment uploaded successfully!')
    except Exception as e:
        conn.rollback()
        flash(f'Error: {str(e)}')
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('teacher'))

@app.route('/submit_assignment/<int:assignment_id>', methods=['POST'])
def submit_assignment(assignment_id):
    if session.get('role') != 'student':
        return redirect(url_for('index'))
    
    try:
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_STUDENT, filename)
            file.save(filepath)
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO submissions (student_id, assignment_id, filename, filepath)
                VALUES (%s, %s, %s, %s)
                """, (
                session['user_id'],
                assignment_id,
                filename,
                filepath
            ))
            
            conn.commit()
            flash('Assignment submitted successfully!')
    except Exception as e:
        conn.rollback()
        flash(f'Error: {str(e)}')
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('student'))

@app.route('/generate_feedback/<int:submission_id>', methods=['POST'])
def generate_feedback(submission_id):
    if session.get('role') != 'teacher':
        return redirect(url_for('index'))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT filepath, filename FROM submissions WHERE id = %s", (submission_id,))
        filepath, filename = cur.fetchone()
        
        ext = filename.split('.')[-1].lower()
        is_handwritten = ext in ['png', 'jpg', 'jpeg']
        
        try:
            if ext == 'pdf':
                text = extract_text_from_pdf(filepath)
            elif ext in ['docx', 'doc']:
                text = extract_text_from_docx(filepath)
            elif is_handwritten:
                text = extract_handwritten_text(filepath)
            else:
                raise ValueError("Unsupported file type")
            
            feedback, marks = generate_feedback_from_text(text, is_handwritten)
            
            cur.execute("""
                UPDATE submissions 
                SET feedback = %s, marks = %s
                WHERE id = %s
                """, (feedback, marks, submission_id))
            
            conn.commit()
            flash('Feedback generated successfully!')
        except Exception as e:
            conn.rollback()
            flash(f'Error processing file: {str(e)}')
        
    except Exception as e:
        flash(f'Database error: {str(e)}')
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('teacher'))

@app.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(os.path.join('uploads', folder), filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/<path:path>')
def catch_all(path):
    flash("The requested URL was not found")
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)