// server.js
require('dotenv').config();
const express = require('express');
const { MongoClient } = require('mongodb');
const cors = require('cors');

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const uri = process.env.MONGO_URI;
const client = new MongoClient(uri);
const dbName = 'Bovine_Classifier';
let db;

async function startServer() {
    try {
        await client.connect();
        db = client.db(dbName);
        console.log('Node.js Server: Successfully connected to MongoDB.');
        app.listen(port, () => {
            console.log(`Node.js data server is running on http://localhost:${port}`);
        });
    } catch (error) {
        console.error('Node.js Server: Failed to connect to MongoDB', error);
        process.exit(1);
    }
}

app.post('/api/saveUpload', async (req, res) => {
    const { browserId, upload } = req.body;
    if (!browserId || !upload) return res.status(400).json({ message: 'Invalid data received.' });
    try {
        const collection = db.collection('website_uploads');
        delete upload.image;
        await collection.updateOne(
            { browserId: browserId },
            { $push: { uploads: upload }, $setOnInsert: { browserId: browserId, firstSeen: new Date() } },
            { upsert: true }
        );
        res.status(201).json({ message: 'Data saved successfully!'});
    } catch (error) {
        console.error('Node.js Server: Failed to save data to MongoDB', error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

app.get('/api/getBotData', async (req, res) => {
    try {
        const collection = db.collection('user_interactions');
        const data = await collection.find({}).sort({ _id: -1 }).toArray();
        res.status(200).json(data);
    } catch (error) {
        console.error('Node.js Server: Failed to fetch bot data', error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

app.get('/api/getWebsiteData', async (req, res) => {
    try {
        const collection = db.collection('website_uploads');
        const data = await collection.find({}).sort({ _id: -1 }).toArray();
        res.status(200).json(data);
    } catch (error) {
        console.error('Node.js Server: Failed to fetch website data', error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

startServer();