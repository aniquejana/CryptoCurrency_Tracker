import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import {
  Box,
  Typography,
  Grid,
  TextField,
  Button,
  Container,
} from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
  root: {
    minHeight: '50vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    background: "linear-gradient(to right, #000000, #434343)",
    padding: theme.spacing(2),
    color: 'white',
  },
  title: {
    marginBottom: theme.spacing(3),
    marginTop: theme.spacing(3),
  },
  info: {
    marginBottom: theme.spacing(3),
    textAlign: 'left',
  },
  infoItem: {
    marginBottom: theme.spacing(2),
    fontSize: '1.2rem',
  },
  form: {
    '& .MuiTextField-root': {
      margin: theme.spacing(1),
      width: '100%',
      '& .MuiOutlinedInput-root': {
        '& fieldset': {
          borderColor: 'white',
        },
        '&:hover fieldset': {
          borderColor: 'white',
        },
        '&.Mui-focused fieldset': {
          borderColor: 'white',
        },
        color: 'white',
      },
      '& label': {
        color: 'white',
      },
      '& label.Mui-focused': {
        color: 'white',
      },
    },
    padding: theme.spacing(3),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
  map: {
    border: 0,
    width: '80%',
    height: '200px',
    marginBottom: theme.spacing(3),
  },
  copyright: {
    textAlign: 'center',
    marginTop: theme.spacing(3),
  },
}));

const ContactForm = () => {
  const classes = useStyles();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
  
    const formActionURL =
      'https://docs.google.com/forms/u/0/d/e/1FAIpQLScNuL2e6K0jlmHBNM-DOOWWCkfN90qpleDWW2_xYuwPyLgxzA/formResponse?pli=1';
    const formDataGoogle = new FormData();
  
    const encodedName = encodeURIComponent(formData.name);
    const encodedEmail = encodeURIComponent(formData.email);
    const encodedSubject = encodeURIComponent(formData.subject);
    const encodedMessage = encodeURIComponent(formData.message);
  
    formDataGoogle.append('entry.223276706', encodedName); 
    formDataGoogle.append('entry.1011578459', encodedEmail); 
    formDataGoogle.append('entry.207203606', encodedSubject); 
    formDataGoogle.append('entry.2087405520', encodedMessage); 
    
    fetch(formActionURL, {
      method: 'POST',
      body: formDataGoogle,
      mode: 'no-cors', // prevent CORS issues
    })
      .then(() => {
        console.log('Form submitted');
        // Reset form data after successful submission
        setFormData({
          name: '',
          email: '',
          subject: '',
          message: '',
        });
      })
      .catch((error) => {
        console.error('Error submitting form', error);
      });
  };

  return (
    <Box className={classes.root}>
      <Container maxWidth="md">
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Box className={classes.info}>
              <Typography variant="h4" className={classes.title}>
                CONTACT
              </Typography>

              <div className={classes.info}>
                <Typography variant="subtitle1">Location:</Typography>
                <iframe
                  src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3684.0228729986484!2d88.47320127349049!3d22.578247832766728!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3a02753269586b03%3A0xfbb4d0c346a81109!2sTechno%20International%20New%20Town!5e0!3m2!1sen!2sin!4v1717174132981!5m2!1sen!2sin"
                  className={classes.map}
                  allowFullScreen=""
                  loading="lazy"
                  referrerPolicy="no-referrer-when-downgrade"
                ></iframe>
              </div>
              <div className={classes.infoItem}>
                <Typography variant="subtitle1" >Email:</Typography>
                <Typography>cointtracker@gmail.com</Typography>
              </div>
              <div className={classes.infoItem}>
                <Typography variant="subtitle1">Call:</Typography>
                <Typography>+91-1234567890</Typography>
              </div>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Box className={classes.form}>
              <form onSubmit={handleSubmit}>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      name="name"
                      label="Your Name"
                      variant="outlined"
                      value={formData.name}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      name="email"
                      label="Your Email"
                      variant="outlined"
                      value={formData.email}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      name="subject"
                      label="Subject"
                      variant="outlined"
                      value={formData.subject}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      name="message"
                      label="Message"
                      variant="outlined"
                      multiline
                      rows={4}
                      value={formData.message}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      color="primary"
                      className={classes.submit}
                    >
                      Send Message
                    </Button>
                  </Grid>
                </Grid>
              </form>
            </Box>
          </Grid>
        </Grid>
        <Box className={classes.copyright}>
          <Typography variant="body2" style={{ fontSize: '1.5rem' }}>
            &copy; {new Date().getFullYear()} DATA PIRATES. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default ContactForm;
