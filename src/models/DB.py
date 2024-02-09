import psycopg2
import os
import re


class DB:
    def __init__(self):
        """
        Initializes a new instance of the DB class.
        """
        self.conn = self._connect_to_db()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self._initial_setup()

    def _initial_setup(self):
        """
        Perform the initial setup of the database by creating necessary tables if they don't exist.

        This method executes SQL queries to create the 'courses', 'skills', and 'course_skills' tables
        if they are not already present in the database. It also logs a message indicating that the
        database setup is complete.
        """
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS courses (id VARCHAR(255) PRIMARY KEY, text TEXT NOT NULL)"
        )
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS skills (id VARCHAR(255) PRIMARY KEY, name VARCHAR(255) NOT NULL, taxonomy VARCHAR(255) NOT NULL)"
        )
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS course_skills (id SERIAL PRIMARY KEY, course_id VARCHAR(255) NOT NULL, skill_id VARCHAR(255) NOT NULL, valid BOOLEAN NOT NULL)"
        )

    def _connect_to_db(self):
        """
        Connects to the PostgreSQL database.

        Returns:
            psycopg2.extensions.connection: The database connection object.

        Raises:
            Exception: If the POSTGRES_PASSWORD environment variable is not set.
        """
        # Connect to db
        if not os.getenv("POSTGRES_PASSWORD"):
            raise Exception("POSTGRES_PASSWORD not set")
        conn = psycopg2.connect(
            f"postgres://postgres:{os.getenv('POSTGRES_PASSWORD')}@postgres:{os.getenv('POSTGRES_PORT')}/postgres?sslmode=disable"
        )
        return conn

    def generate_cleaned_text_hash(self, text: str):
        """
        Generates a cleaned text hash.

        Args:
            text (str): The input text to be cleaned and hashed.

        Returns:
            str: The hashed representation of the cleaned text.
        """
        text = text.lower()
        text = re.sub(r"\W+", "", text)
        return str(hash(text))

    def update_course_skills(
        self, course: str, validationResults: list, course_id: str = None
    ) -> str:
        # insert course
        # Generate hash of course
        if course_id is None:
            course_id = str(self.generate_cleaned_text_hash(course))

        # Check if course already exists
        self.cursor.execute("SELECT id FROM courses WHERE id = %s", (course_id,))
        if self.cursor.fetchone() is None:
            # Insert course if it doesn't exist
            self.cursor.execute(
                "INSERT INTO courses (id, text) VALUES (%s, %s)", (course_id, course)
            )
        else:
            # Delete old relations
            self.cursor.execute(
                "DELETE FROM course_skills WHERE course_id = %s", (course_id,)
            )

        # insert skills
        for skill in validationResults:
            skill_id = skill["uri"]
            self.cursor.execute("SELECT id FROM skills WHERE id = %s", (skill_id,))
            if self.cursor.fetchone() is None:
                self.cursor.execute(
                    "INSERT INTO skills (id, name, taxonomy) VALUES (%s, %s, %s)",
                    (skill_id, skill["title"], skill["taxonomy"]),
                )

            # insert relation
            self.cursor.execute(
                "SELECT id FROM course_skills WHERE course_id = %s AND skill_id = %s",
                (course_id, skill_id),
            )
            # Insert relation if it doesn't exist
            if self.cursor.fetchone() is None:
                self.cursor.execute(
                    "INSERT INTO course_skills (course_id, skill_id, valid) VALUES (%s, %s, %s)",
                    (course_id, skill_id, skill["valid"]),
                )
            # Update relation
            else:
                self.cursor.execute(
                    "UPDATE course_skills SET valid = %s WHERE course_id = %s AND skill_id = %s",
                    (skill["valid"], course_id, skill_id),
                )
        
        return course_id

    def get_course_skills(self):
        """
        Retrieves the course skills from the database.

        Returns:
            A list of tuples containing the course text, skill id, and validity.
        """
        self.cursor.execute(
            "SELECT d.text, s.id, ds.valid FROM courses d, skills s, course_skills ds WHERE d.id = ds.course_id AND s.id = ds.skill_id"
        )
        return self.cursor.fetchall()
