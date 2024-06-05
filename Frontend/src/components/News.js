import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    marginTop: 25,
    marginLeft: 15,
    marginRight: 15,
  },
  paper: {
    padding: theme.spacing(3),
    textAlign: 'center',
    color: 'white', 
    background: 'linear-gradient(to bottom right, rgba(255, 255, 255, 0.1), rgba(0, 0, 0, 0.3))', 
    backdropFilter: 'blur(5px)', 
    boxShadow: '0 8px 32px 0 rgba(40, 40, 40, 0.6)',
    borderRadius: '10px', 
  },
}));

const News = () => {
  const classes = useStyles();
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await fetch('https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=40640aaa68e04e7f8131a4ed4eb914a4');
        if (!response.ok) {
          throw new Error('Failed to fetch news');
        }
        const data = await response.json();
        setArticles(data.articles.slice(0,6));
      } catch (error) {
        console.error('Error fetching news:', error);
      }
    };

    fetchNews();
  }, []);

  return (
    <div className={classes.root}>
      <Grid container spacing={3}>
        {articles.map((article, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Paper className={classes.paper}>
              <a href={article.url} target="_blank" rel="noopener noreferrer">
                <img src={article.urlToImage} alt={article.title} style={{ maxWidth: '100%', marginBottom: '10px' }} />
                <h2>{article.title}</h2>
              </a>
              <p>{article.description}</p>
              <p>Published on: {new Date(article.publishedAt).toDateString()}</p>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </div>
  );
};

export default News;
