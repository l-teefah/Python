import imaplib
import email
import pandas as pd
from email.header import decode_header
from datetime import datetime, timedelta
import re

class YahooJobExtractor:
    def __init__(self, email_address, password):
        self.email_address = email_address
        self.password = password
        self.mail = None
        
    def connect(self):
        """Connect to Yahoo Mail via IMAP"""
        try:
            self.mail = imaplib.IMAP4_SSL("imap.mail.yahoo.com", 993)
            self.mail.login(self.email_address, self.password)
            print("Successfully connected to Yahoo Mail")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def decode_subject(self, subject):
        """Decode email subject"""
        if subject:
            decoded_parts = decode_header(subject)
            decoded_subject = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    decoded_subject += part.decode(encoding or 'utf-8')
                else:
                    decoded_subject += part
            return decoded_subject
        return ""
    
    def extract_job_emails(self, start_date=None, end_date=None, search_terms=None):
        """
        Extract job application emails within date range
        
        Args:
            start_date (str): Start date in format 'DD-Mon-YYYY' (e.g., '01-Aug-2023')
            end_date (str): End date in format 'DD-Mon-YYYY' (e.g., '31-Mar-2025')
            search_terms (list): Keywords to search for (default job-related terms)
        """
        if not self.mail:
            print("Not connected to email")
            return []
        
        # Default search terms for job applications
        if search_terms is None:
            search_terms = [
                'application', 'apply', 'position', 'job', 'career', 
                'opportunity', 'interview', 'resume', 'cv', 'hiring',
                'stage', 'unfortunately', 'thank you for applying',
                'we received your application', 'application received'
            ]
        
        try:
            # Select the inbox folder
            self.mail.select('"INBOX"')
            
            # Start with basic search - Yahoo IMAP is picky about complex searches
            search_criteria = []
            
            # Add date constraints if provided (Yahoo format: DD-Mon-YYYY)
            if start_date:
                search_criteria.append(f'SINCE {start_date}')
            if end_date:
                search_criteria.append(f'BEFORE {end_date}')
            
            # Build search query
            if search_criteria:
                search_query = ' '.join(search_criteria)
            else:
                search_query = 'ALL'
            
            print(f"Searching with query: {search_query}")
            status, messages = self.mail.search(None, search_query)
            
            if status != 'OK':
                print(f"Date search failed, trying without date constraints...")
                # Fallback: search all emails
                status, messages = self.mail.search(None, 'ALL')
                if status != 'OK':
                    print("All searches failed")
                    return []
            
            email_ids = messages[0].split()
            print(f"Found {len(email_ids)} emails to check")
            
            job_applications = []
            
            for i, email_id in enumerate(email_ids):
                if i % 10 == 0:  # Progress indicator
                    print(f"Processing email {i+1}/{len(email_ids)}")
                
                try:
                    # Fetch email headers only
                    status, msg_data = self.mail.fetch(email_id, '(RFC822.HEADER)')
                    
                    if status == 'OK':
                        email_message = email.message_from_bytes(msg_data[0][1])
                        
                        # Extract relevant information
                        subject = self.decode_subject(email_message.get('Subject', ''))
                        from_email = email_message.get('From', '')
                        to_email = email_message.get('To', '')
                        date_str = email_message.get('Date', '')
                        
                        # Check if this email is job-related
                        if self.is_job_related(subject, from_email, to_email, search_terms):
                            # Parse date
                            try:
                                parsed_date = email.utils.parsedate_to_datetime(date_str)
                                formatted_date = parsed_date.strftime('%Y-%m-%d')
                                
                                # Filter by date range if specified
                                if start_date or end_date:
                                    if not self.is_in_date_range(parsed_date, start_date, end_date):
                                        continue
                                        
                            except:
                                formatted_date = date_str
                            
                            # Extract company name from email domain or subject
                            company = self.extract_company_name(from_email, subject)
                            
                            # Extract position from subject
                            position = self.extract_position(subject)
                            
                            job_applications.append({
                                'Date': formatted_date,
                                'Subject': subject,
                                'Company': company,
                                'Position': position,
                                'From_Email': from_email,
                                'To_Email': to_email,
                                'Email_ID': email_id.decode()
                            })
                        
                except Exception as e:
                    print(f"Error processing email {email_id}: {e}")
                    continue
            
            print(f"Found {len(job_applications)} job-related emails")
            return job_applications
            
        except Exception as e:
            print(f"Error extracting emails: {e}")
            return []
    
    def is_job_related(self, subject, from_email, to_email, search_terms):
        """Check if an email is job-related based on subject and sender"""
        if not subject:
            return False
        
        subject_lower = subject.lower()
        from_email_lower = from_email.lower() if from_email else ""
        
        # Check if any search terms appear in subject
        for term in search_terms:
            if term.lower() in subject_lower:
                return True
        
        # Check for job-related domains in sender
        job_domains = ['careers', 'jobs', 'hr', 'recruiting', 'talent', 'noreply', 'workday', 'greenhouse', 'lever']
        for domain in job_domains:
            if domain in from_email_lower:
                return True
        
        # Check for common job-related phrases
        job_phrases = [
            'thank you for your application',
            'we received your application',
            'application received',
            'thank you for applying',
            'your application for',
            'application confirmation',
            'application status',
            'next steps',
            'move forward',
            'unfortunately',
            'not selected',
            'next stage'
        ]
        
        for phrase in job_phrases:
            if phrase in subject_lower:
                return True
        
        return False
    
    def is_in_date_range(self, email_date, start_date, end_date):
        """Check if email date is within specified range"""
        try:
            if start_date:
                start_dt = datetime.strptime(start_date, '%d-%b-%Y')
                if email_date < start_dt:
                    return False
            
            if end_date:
                end_dt = datetime.strptime(end_date, '%d-%b-%Y')
                if email_date > end_dt:
                    return False
            
            return True
        except:
            return True  # If date parsing fails, include the email
    
    def extract_company_name(self, email_address, subject):
        """Extract company name from email address or subject"""
        if '@' in email_address:
            domain = email_address.split('@')[-1].split('.')[0]
            # Clean up common domain parts
            domain = domain.replace('careers', '').replace('jobs', '').replace('hr', '').replace('noreply', '')
            if domain and len(domain) > 1:
                return domain.title()
        
        # Try to extract from subject
        subject_words = subject.split()
        for word in subject_words:
            if len(word) > 3 and word.isalpha() and word[0].isupper():
                return word
        
        return "Unknown"
    
    def extract_position(self, subject):
        """Extract position title from subject line"""
        # Common patterns for job positions
        position_patterns = [
            r'application for (.+?)(?:\s*-|\s*at|\s*\||\s*$)',
            r'applying for (.+?)(?:\s*-|\s*at|\s*\||\s*$)',
            r'(.+?) position',
            r'(.+?) role',
            r'(.+?) opportunity',
            r'your application for (.+?)(?:\s*-|\s*at|\s*\||\s*$)',
            r'(.+?) - application',
            r'(.+?) application'
        ]
        
        subject_lower = subject.lower()
        for pattern in position_patterns:
            match = re.search(pattern, subject_lower, re.IGNORECASE)
            if match:
                position = match.group(1).strip()
                # Clean up the position title
                position = re.sub(r'^(the|a|an)\s+', '', position, flags=re.IGNORECASE)
                # Remove common suffixes
                position = re.sub(r'\s*(position|role|job|application)$', '', position, flags=re.IGNORECASE)
                if len(position) > 2:
                    return position.title()
        
        # If no pattern matches, return the subject (might contain position info)
        return subject
    
    def generate_summary(self, job_applications):
        """Generate summary statistics"""
        if not job_applications:
            return "No job applications found"
        
        df = pd.DataFrame(job_applications)
        
        summary = {
            'Total Applications': len(job_applications),
            'Unique Companies': df['Company'].nunique(),
            'Unique Positions': df['Position'].nunique(),
            'Date Range': f"{df['Date'].min()} to {df['Date'].max()}"
        }
        
        return summary
    
    def export_to_excel(self, job_applications, filename='job_applications.xlsx'):
        """Export data to Excel file"""
        if not job_applications:
            print("No data to export")
            return
        
        df = pd.DataFrame(job_applications)
        
        # Create summary sheet
        summary = self.generate_summary(job_applications)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Job Applications', index=False)
            
            # Write summary
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write position counts
            position_counts = df['Position'].value_counts().reset_index()
            position_counts.columns = ['Position', 'Count']
            position_counts.to_excel(writer, sheet_name='Position Counts', index=False)
            
            # Write company counts
            company_counts = df['Company'].value_counts().reset_index()
            company_counts.columns = ['Company', 'Count']
            company_counts.to_excel(writer, sheet_name='Company Counts', index=False)
        
        print(f"Data exported to {filename}")
        print(f"Summary: {summary}")
    
    def close(self):
        """Close the email connection"""
        if self.mail:
            self.mail.close()
            self.mail.logout()

# Usage example
def main():
    # Configuration - UPDATE THESE WITH YOUR DETAILS
    EMAIL = "lateefah_yusuf@yahoo.com"
    PASSWORD = "chzx aodx uekf uosu"  # Use App Password, not regular password
    
    # Date range (optional) - Format: DD-Mon-YYYY
    START_DATE = "01-Aug-2023"  
    END_DATE = "31-Mar-2025"    
    
    # Custom search terms (optional)
    SEARCH_TERMS = [
        'application', 'apply', 'position', 'job', 'career',
        'opportunity', 'interview', 'resume', 'cv', 'hiring',
        'stage', 'unfortunately', 'thank you for applying'
    ]
    
    print("Starting Yahoo Mail job application extraction...")
    print(f"Email: {EMAIL}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    # Initialize extractor
    extractor = YahooJobExtractor(EMAIL, PASSWORD)
    
    # Connect and extract
    if extractor.connect():
        try:
            job_applications = extractor.extract_job_emails(
                start_date=START_DATE,
                end_date=END_DATE,
                search_terms=SEARCH_TERMS
            )
            
            if job_applications:
                # Export to Excel
                extractor.export_to_excel(job_applications, 'my_job_applications.xlsx')
                print("✅ Extraction completed successfully!")
            else:
                print("❌ No job applications found. Try:")
                print("1. Checking if emails are in different folders")
                print("2. Expanding the date range")
                print("3. Adding more search terms")
                print("4. Checking both INBOX and Sent folders")
        
        except Exception as e:
            print(f"Error during extraction: {e}")
        
        finally:
            # Close connection
            extractor.close()
    else:
        print("❌ Failed to connect to Yahoo Mail")
        print("Make sure you're using an App Password, not your regular password")

if __name__ == "__main__":
    main()
