'use client';

import {
  Box,
  Grid,
  Typography,
  Button,
  Rating,
  Card,
  CardContent,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
} from "@mui/material";
import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder";
import { useState } from "react";
import SizeRecommendationModal from "./modal";

const product = {
  title: "Women's Mid Rise '94 Baggy Fit Jeans",
  price: "$99",
  originalPrice: "$119",
  images: ["/jeans1.jpg", "/jeans2.jpg", "/jeans3.jpg"],
  rating: 4.2,
  sizes: ["XS", "S", "M", "L", "XL", "XXL", "3XL"],
  unavailableSizes: ["XL", "XXL", "3XL"],
  fit: "Baggy Fit",
  rise: "Mid Rise",
  length: "Full Length",
  fabric: "Cotton",
};

export default function ProductPage() {
  const [selectedSize, setSelectedSize] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const handleOpen = () => setModalOpen(true);
  const handleClose = () => setModalOpen(false);

  return (
    <Box sx={{ maxWidth: "1200px", mx: "auto", p: 4 }}>
      <Grid container spacing={4}>
        {/* Image Section */}
        <Grid item xs={12} md={6}>
          <Grid container spacing={2}>
            {product.images.map((src, idx) => (
              <Grid item xs={6} key={idx}>
                <Box
                  component="img"
                  src={src}
                  alt={`Product ${idx + 1}`}
                  sx={{ width: "100%", borderRadius: 2, boxShadow: 2 }}
                />
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Details Section */}
        <Grid item xs={12} md={6}>
          <Typography variant="h5" fontWeight={600}>
            {product.title}
          </Typography>

          {/* Price */}
          <Box mt={1} display="flex" gap={2} alignItems="center">
            <Typography variant="h6" color="error">
              {product.price}
            </Typography>
            <Typography variant="body1" sx={{ textDecoration: "line-through" }} color="text.secondary">
              {product.originalPrice}
            </Typography>
          </Box>

          {/* Rating */}
          <Box mt={1} display="flex" alignItems="center" gap={1}>
            <Rating value={product.rating} readOnly precision={0.1} />
            <Typography variant="body2" color="text.secondary">
              {product.rating}/5
            </Typography>
          </Box>

          {/* Size Selector */}
          <Box mt={3}>
            <Typography fontWeight={500}>Select Size</Typography>
            <ToggleButtonGroup
              exclusive
              value={selectedSize}
              onChange={(e, newSize) => setSelectedSize(newSize)}
              sx={{ mt: 1, flexWrap: "wrap" }}
            >
              {product.sizes.map((size) => (
                <ToggleButton
                  key={size}
                  value={size}
                  disabled={product.unavailableSizes.includes(size)}
                  sx={{ minWidth: 56 }}
                >
                  {size}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            {/* Recommend My Size Button */}
            <Box mt={2}>
            <Button variant="outlined" onClick={() => setModalOpen(true)}>
              Recommend My Size
            </Button>
            </Box>
          </Box>

          {/* Actions */}
          <Box mt={3} display="flex" gap={2}>
            <Button
              variant="contained"
              color="primary"
              sx={{ flex: 1, textTransform: "none", fontWeight: 500 }}
            >
              Add to Bag
            </Button>
            <IconButton color="primary" aria-label="wishlist">
              <FavoriteBorderIcon />
            </IconButton>
          </Box>

          {/* Product Info */}
          <Card variant="outlined" sx={{ mt: 4 }}>
            <CardContent>
              <Typography variant="subtitle2"><strong>Fit:</strong> {product.fit}</Typography>
              <Typography variant="subtitle2"><strong>Rise:</strong> {product.rise}</Typography>
              <Typography variant="subtitle2"><strong>Length:</strong> {product.length}</Typography>
              <Typography variant="subtitle2"><strong>Fabric:</strong> {product.fabric}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>


    {/* <SizeRecommendationModal open={modalOpen} onClose={() => setModalOpen(false)} /> */}
    <SizeRecommendationModal
      open={modalOpen}
      onClose={handleClose}
      onSelectSize={(size) => {
        setSelectedSize(size);
        handleClose();
      }}
    />
    </Box>
  );
}

export const dynamic = 'error';