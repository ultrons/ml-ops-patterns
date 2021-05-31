import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

        setuptools.setup(
                name="view_demo",
                version="0.0.1",
                author="Vaibhav Singh",
                author_email="sivaibhav@google.com",
                description="MLOPs case study with Vertex AI",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/xx/yy",
            classifiers=[
                        "Programming Language :: Python :: 3",
                        "License :: OSI Approved :: MIT License",
                        "Operating System :: OS Independent",

            ],
                package_dir={"": "."},
                packages=setuptools.find_packages(exclude=('view_demo/research')),
                python_requires=">=3.6",

        )
